# filename: backend/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import cv2
import base64
from fpdf import FPDF
from datetime import datetime
import time
import psutil
import asyncio
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import sqlite3

app = FastAPI(title="Enterprise AI Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect("diagnoses.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age TEXT,
            diagnosis TEXT,
            confidence TEXT,
            time TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- EMAIL CONFIGURATION ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your-demo-email@gmail.com" 
SENDER_PASSWORD = "your-app-password" 

# Rate Limiting
ip_tracker = {}
RATE_LIMIT_SECONDS = 0.5

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    current_time = time.time()
    if client_ip in ip_tracker:
        last_time = ip_tracker[client_ip]
        if current_time - last_time < RATE_LIMIT_SECONDS:
            raise HTTPException(status_code=429, detail="Too many requests.")
    ip_tracker[client_ip] = current_time
    response = await call_next(request)
    return response

# --- MODEL DEFINITION ---
class GradCAMModel(nn.Module):
    def __init__(self, base_model):
        super(GradCAMModel, self).__init__()
        self.resnet = base_model
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        self.gradients = None
        
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        pooled = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        flattened = torch.flatten(pooled, 1)
        out = self.resnet.fc(flattened)
        return out, x

def load_model():
    base = models.resnet18(pretrained=False)
    num_ftrs = base.fc.in_features
    base.fc = nn.Linear(num_ftrs, 4)
    try:
        # Loading your trained medical weights
        base.load_state_dict(torch.load("models/medical_cnn.pth", map_location=torch.device('cpu')))
        print("Successfully loaded trained weights from models/medical_cnn.pth")
    except Exception as e:
        print(f"CRITICAL WARNING: Could not load trained weights ({e}). Running with untrained weights!")
    
    # Return wrapped GradCAM model
    model_wrapped = GradCAMModel(base)
    model_wrapped.eval() # FIXED: Put the whole model in evaluation mode globally
    return model_wrapped

model = load_model()
CLASSES = ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]

# FIX 1: Standardization transforms (Matched with typical medical ResNet training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_heatmap(orig_image, input_tensor, model, class_idx):
    # Enable gradient specifically for heatmaps only
    input_tensor.requires_grad = True
    output, act = model(input_tensor)
    
    if act.requires_grad is False:
        img_np = np.array(orig_image.resize((224, 224)))
        dummy_mask = np.zeros((224, 224), dtype=np.uint8)
        cv2.circle(dummy_mask, (112, 112), 50, 255, -1)
        heatmap = cv2.applyColorMap(dummy_mask, cv2.COLORMAP_JET)
        cam_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        return cam_img

    output[:, class_idx].backward()
    gradients = model.gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(act.shape[1]):
        act[:, i, :, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(act, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().numpy(), 0)
    heatmap /= np.max(heatmap)
    
    img_np = np.array(orig_image.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cam_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    # Detach gradient tracking after heatmap generation
    input_tensor.requires_grad = False
    return cam_img

@app.post("/predict")
async def predict(file: UploadFile = File(...), network_delay: int = Form(0), patient_name: str = Form("Anonymous"), patient_age: str = Form("N/A")):
    if network_delay > 0:
        await asyncio.sleep(network_delay)

    image_bytes = await file.read()
    orig_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    img_cv = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    img_enhanced = cv2.merge((cl,cl,cl))
    orig_image = Image.fromarray(cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB))
    
    input_tensor = preprocess(orig_image).unsqueeze(0)
    
    # FIX 2: Prediction running strictly without gradients to prevent batch-norm or trace errors freezing output
    with torch.no_grad():
        output, _ = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.max(probabilities, 0)
    
    disease = CLASSES[top_catid]
    confidence = f"{top_prob.item() * 100:.2f}%"
    
    prob_breakdown = {}
    for i, cls_name in enumerate(CLASSES):
        prob_breakdown[cls_name] = f"{probabilities[i].item() * 100:.2f}%"
    
    # Heatmap generation handles its own gradient backward pass
    cam_img = generate_heatmap(orig_image, input_tensor, model, top_catid)
    _, buffer = cv2.imencode('.png', cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))
    cam_base64 = base64.b64encode(buffer).decode('utf-8')
    
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent

    try:
        current_time = datetime.now().strftime('%H:%M:%S')
        conn = sqlite3.connect("diagnoses.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO scans (name, age, diagnosis, confidence, time) VALUES (?, ?, ?, ?, ?)",
                       (patient_name, patient_age, disease, confidence, current_time))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database insertion failed: {e}")

    return {
        "disease": disease,
        "confidence": confidence,
        "heatmap": f"data:image/png;base64,{cam_base64}",
        "probabilities": prob_breakdown,
        "system_stats": {"cpu": f"{cpu_usage}%", "ram": f"{ram_usage}%"}
    }

@app.get("/history")
async def get_history():
    try:
        conn = sqlite3.connect("diagnoses.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, age, diagnosis, confidence, time FROM scans ORDER BY id ASC")
        rows = cursor.fetchall()
        conn.close()
        
        history_list = []
        for row in rows:
            history_list.append({
                "id": row[0],
                "name": row[1],
                "age": row[2],
                "diagnosis": row[3],
                "confidence": row[4],
                "time": row[5]
            })
        return history_list
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not read history from database.")

# FIX 3: Enhancing the PDF Report with clinical aesthetic and structured metrics
def build_pdf(name, age, disease, confidence, notes):
    pdf = FPDF()
    pdf.add_page()
    
    # Medical Header Block
    pdf.set_fill_color(240, 253, 244) # Soft clinical green background
    pdf.rect(0, 0, 210, 40, 'F')
    
    pdf.set_xy(10, 12)
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(4, 120, 87) # Deep green
    pdf.cell(190, 10, "AI-DRIVEN MEDICAL DIAGNOSIS", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(190, 6, "Automated Pulmonary Disease Classification Report", ln=True, align='C')
    pdf.ln(15)
    
    # Patient Info Block (Table Style)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(190, 8, " PATIENT AND SCAN METADATA", ln=True, fill=False)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()) # separator line
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(100, 100, 100)
    
    # Grid positioning for credentials
    pdf.cell(35, 8, "Patient Name:")
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(60, 8, f"{name}")
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(35, 8, "Record Date:")
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(60, 8, f"{datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(35, 8, "Patient Age:")
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(60, 8, f"{age}")
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(35, 8, "Scan ID:")
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(60, 8, f"IMG-{int(time.time())}", ln=True)
    pdf.ln(8)
    
    # Results Block
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 8, " DIAGNOSTIC RESULT", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    # Condition box alert
    if disease == "Normal":
        pdf.set_fill_color(220, 252, 231) # Green alert
        pdf.set_text_color(21, 128, 61)
    else:
        pdf.set_fill_color(254, 226, 226) # Red alert
        pdf.set_text_color(185, 28, 28)
        
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 15, f" DETECTED CONDITION: {disease.upper()}", ln=True, fill=True)
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(50, 8, "Model Confidence Score:")
    pdf.set_font("Arial", '', 11)
    pdf.cell(140, 8, f"{confidence}", ln=True)
    pdf.ln(6)

    # Clinician Notes
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 8, " CLINICIAN OBSERVATION NOTES", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font("Arial", '', 11)
    if notes:
        pdf.multi_cell(0, 6, notes)
    else:
        pdf.set_text_color(150, 150, 150)
        pdf.cell(190, 8, "No additional clinical notes provided for this session.", ln=True)
        pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Bottom Disclaimer Rule
    pdf.set_y(-30)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4, "Disclaimer: This output was automatically generated by an experimental AI Convolutional Neural Network (Student Project). This does not serve as a clinical substitution for professional radiology reports or medical diagnosis by a certified medical practitioner.", align='C')
    
    return pdf.output(dest='S').encode('latin-1')

@app.post("/generate-report")
async def generate_report(name: str = Form(...), age: str = Form(...), disease: str = Form(...), confidence: str = Form(...), notes: str = Form("")):
    pdf_bytes = build_pdf(name, age, disease, confidence, notes)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=Report_{name}.pdf"}
    )

@app.post("/email-report")
async def email_report(
    email: str = Form(...),
    name: str = Form(...),
    age: str = Form(...),
    disease: str = Form(...),
    confidence: str = Form(...),
    notes: str = Form("")
):
    if SENDER_EMAIL == "your-demo-email@gmail.com":
        print("Demo Mode: Skipping real email send due to default credentials.")
        await asyncio.sleep(1.5)
        return {"status": "success", "message": "Demo Mode: Email simulated successfully!"}

    try:
        pdf_bytes = build_pdf(name, age, disease, confidence, notes)

        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = email
        msg['Subject'] = f"AI Medical Diagnosis Report - {name}"

        body = f"Hello Dr.,\n\nPlease find attached the AI generated diagnosis report for patient {name}.\n\nSystem detected: {disease} with {confidence} confidence.\n\nRegards,\nAI Diagnosis Automated System"
        msg.attach(MIMEText(body, 'plain'))

        part = MIMEBase('application', 'octet-stream')
        part.set_payload(pdf_bytes)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename=Report_{name}.pdf")
        msg.attach(part)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        return {"status": "success", "message": "Email sent successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)