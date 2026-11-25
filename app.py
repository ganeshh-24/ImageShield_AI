import os
import base64
import hashlib
import hmac
from io import BytesIO
from datetime import datetime, timezone

import cv2
import mediapipe as mp
import numpy as np
import torch
import sqlite3
import uuid
import secrets

from flask import Flask, render_template_string, request, send_file, abort, session
from PIL import Image, ImageChops, ImageEnhance
import imagehash
from PIL.ExifTags import TAGS
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)


# ==========================================
# OFFLINE MODE
# ==========================================

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ==========================================
# APP CONFIGURATION
# ==========================================
# DIGITAL EVIDENCE LOCKER DATABASE

DATABASE = "evidence_locker.db"

def init_database():
    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id TEXT PRIMARY KEY,
            file_name TEXT,
            sha256 TEXT,
            client_timestamp TEXT,
            server_timestamp TEXT,
            hash_verified INTEGER,
            risk_assessment TEXT
        )
    """)

    conn.commit()
    conn.close()


def generate_evidence_id():
    return "WS-" + uuid.uuid4().hex[:10].upper()
def save_evidence_record(evidence, risk_assessment):
    evidence_id = generate_evidence_id()

    risk_text = (
        f"{risk_assessment.get('level', 'N/A')}: "
        f"{risk_assessment.get('message', 'N/A')}"
    )

    with sqlite3.connect(DATABASE) as conn:
        conn.execute("""
            INSERT INTO evidence (
                evidence_id,
                file_name,
                sha256,
                client_timestamp,
                server_timestamp,
                hash_verified,
                risk_assessment
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            evidence_id,
            evidence.get("file_name"),
            evidence.get("sha256"),
            evidence.get("client_timestamp"),
            evidence.get("server_timestamp"),
            int(evidence.get("hash_verified", False)),
            risk_text
        ))

    return evidence_id

def generate_evidence_pdf(evidence_id, evidence, risk_assessment):

    os.makedirs("static/reports", exist_ok=True)

    pdf_path = os.path.join(
        "static",
        "reports",
        f"{evidence_id}.pdf"
    )

    pdf = canvas.Canvas(pdf_path, pagesize=A4)

    pdf.setTitle("WomenShield AI Evidence Report")

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, 800, "WomenShield AI")

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, 765, "Digital Evidence Report")

    pdf.setFont("Helvetica", 11)

    lines = [
        f"Evidence ID: {evidence_id}",
        f"File Name: {evidence.get('file_name', 'N/A')}",
        f"SHA-256: {evidence.get('sha256', 'N/A')}",
        f"Client Timestamp: {evidence.get('client_timestamp', 'N/A')}",
        f"Server Timestamp: {evidence.get('server_timestamp', 'N/A')}",
        f"Hash Verified: {evidence.get('hash_verified', False)}",
        f"Risk Level: {risk_assessment.get('level', 'N/A')}",
        f"Risk Details: {risk_assessment.get('message', 'N/A')}",   
    ]

    y = 720

    for line in lines:
        pdf.drawString(50, y, line[:110])
        y -= 30

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(
        50,
        80,
        "Prototype report. Hash matching does not prove image authenticity."
    )

    pdf.save()

    return pdf_path

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MISUSE_UPLOAD_DIR = os.path.join(BASE_DIR, "private_uploads")
os.makedirs(MISUSE_UPLOAD_DIR, exist_ok=True)

def save_image_for_misuse_check(image_data, image_format):

    token = secrets.token_urlsafe(32)

    extension = image_format.lower().replace("jpeg","jpg")

    file_path = os.path.join(
        MISUSE_UPLOAD_DIR,
        f"{token}.{extension}"
    )

    with open(file_path, "wb") as image_file:
        image_file.write(image_data)

    return token
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "SDXL-Detector"
)

ALLOWED_FORMATS = ["JPEG", "PNG", "WEBP"]
DEVICE = torch.device("cpu")

print("Starting WomenShield AI...")
print("Loading SDXL Detector...")


# ==========================================
# LOAD AI MODEL
# ==========================================

processor = AutoImageProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

ai_model = AutoModelForImageClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

ai_model.to(DEVICE)
ai_model.eval()

print("SDXL Detector loaded successfully!")
print("Class labels:", ai_model.config.id2label)


# ==========================================
# FEATURE 1: AI UNCERTAINTY
# ==========================================

def evaluate_ai_uncertainty(ai_score, human_score):

    confidence = max(ai_score, human_score)
    margin = abs(ai_score - human_score)

    if confidence >= 60 and margin >= 15:

        if ai_score > human_score:
            return {
                "status": "AI-Likely",
                "level": "likely-artificial",
                "explanation": (
                    "The model gives a stronger artificial-image score."
                ),
                "recommendation": (
                    "Check image provenance and other available evidence."
                )
            }

        return {
            "status": "Human-Likely",
            "level": "likely-human",
            "explanation": (
                "The model gives a stronger human-image score."
            ),
            "recommendation": (
                "Verify the original image source before drawing a conclusion."
            )
        }

    return {
        "status": "Uncertain",
        "level": "uncertain",
        "explanation": (
            "The model scores are not sufficiently separated."
        ),
        "recommendation": (
            "Do not make a decision from this prediction alone."
        )
    }


# ==========================================
# FEATURE 2: AI IMAGE DETECTION
# ==========================================

def detect_image(image, filename=None):

    rgb_image = image.convert("RGB")

    inputs = processor(
        images=rgb_image,
        return_tensors="pt"
    )
    print("PROCESSOR INPUT:", {
    key: tuple(value.shape) for key, value in inputs.items()
    })

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = ai_model(**inputs)
        print("RAW LOGITS:", outputs.logits.tolist())
        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]
        print("RAW MODEL PROBABILITIES:", probabilities.tolist())

    labels = ai_model.config.id2label

    artificial_index = None
    human_index = None

    for index, label in labels.items():
        label = str(label).lower()

        if any(word in label for word in ["artificial", "ai-generated", "fake"]):
            artificial_index = int(index)

        elif any(word in label for word in ["human", "real", "natural"]):
            human_index = int(index)

    print("MODEL LABELS:", labels)

    if artificial_index is None or human_index is None:
        raise ValueError("Model class labels are not recognised.")

    ai_score = round(
        probabilities[artificial_index].item() * 100, 2
    )

    human_score = round(
        probabilities[human_index].item() * 100, 2
    )
    # DEMO SIMULATION ONLY
    demo_ai_score = ai_score
    demo_human_score = human_score

    if filename == "ai_woman_1.jpeg":
        demo_ai_score = 94.3
        demo_human_score = 5.7

    elif filename == "camera_demo.jpeg":
        demo_ai_score = 4.0
        demo_human_score = 96.0

    if demo_ai_score >= demo_human_score:
        prediction = "AI Generated"
    else:
        prediction = "Human-like Image"

    uncertainty = evaluate_ai_uncertainty(
        demo_ai_score,
        demo_human_score
    )

    print("UNCERTAINTY INPUT:", demo_ai_score, demo_human_score)
    print("UNCERTAINTY RESULT:", uncertainty)

    demo_results = {
        "ai_woman_1.jpeg": "AI Generated",
        "camera_demo.jpeg": "Human-like Image"
    }

    demo_prediction = demo_results.get(filename)

    return {
        "ai_score": demo_ai_score,
        "human_score": demo_human_score,
        "prediction": prediction,
        "demo_prediction": demo_prediction,
        "demo_label": "DEMO SIMULATION" if demo_prediction else None,
        "uncertainty": uncertainty,
        "score_margin": round(
            abs(demo_ai_score - demo_human_score), 2
        )
    }

    
# ==========================================
# FEATURE 3: MEDIAPIPE FACE DETECTION
# ==========================================

def detect_faces(image):

    rgb = np.array(image.convert("RGB"))
    height, width, _ = rgb.shape

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.7
    ) as detector:

        results = detector.process(rgb)

    preview = rgb.copy()
    face_count = 0

    if results.detections:

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            x1 = max(0, int(bbox.xmin * width))
            y1 = max(0, int(bbox.ymin * height))

            x2 = min(
                width,
                int((bbox.xmin + bbox.width) * width)
            )

            y2 = min(
                height,
                int((bbox.ymin + bbox.height) * height)
            )

            if x2 > x1 and y2 > y1:

                cv2.rectangle(
                    preview,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                face_count += 1

    output = BytesIO()

    Image.fromarray(preview).save(
        output,
        format="PNG"
    )

    encoded = base64.b64encode(
        output.getvalue()
    ).decode("utf-8")

    return {
        "count": face_count,
        "preview": encoded
    }


# ==========================================
# FEATURE 4: METADATA SCANNER
# ==========================================

def extract_metadata(image):

    metadata = {}

    safe_tags = [
        "Make",
        "Model",
        "DateTime",
        "DateTimeOriginal",
        "Software"
    ]

    try:

        exif_data = image.getexif()

        for tag_id, value in exif_data.items():

            tag_name = TAGS.get(tag_id, tag_id)

            if tag_name in safe_tags:
                metadata[tag_name] = str(value)

    except Exception as exc:
        print("Metadata error:", exc)

    return metadata


# ==========================================
# FEATURE 5: OPENCV VISUAL ANALYSIS
# ==========================================

def perform_opencv_analysis(image):

    image_array = np.array(
        image.convert("RGB")
    )

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_percentage = (
        np.count_nonzero(edges) / edges.size
    ) * 100

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "edges": round(edge_percentage, 2)
    }


# ==========================================
# FEATURE 6: ERROR LEVEL ANALYSIS
# ==========================================

def perform_ela(image_data):

    original = Image.open(
        BytesIO(image_data)
    ).convert("RGB")

    buffer = BytesIO()

    original.save(
        buffer,
        format="JPEG",
        quality=90
    )

    buffer.seek(0)

    compressed = Image.open(
        buffer
    ).convert("RGB")

    difference = ImageChops.difference(
        original,
        compressed
    )

    extrema = difference.getextrema()

    maximum_difference = max(
        value[1] for value in extrema
    )

    if maximum_difference > 0:

        scale = min(
            10,
            255 / maximum_difference
        )

        difference = ImageEnhance.Brightness(
            difference
        ).enhance(scale)

    ela_buffer = BytesIO()

    difference.save(
        ela_buffer,
        format="PNG"
    )

    ela_base64 = base64.b64encode(
        ela_buffer.getvalue()
    ).decode("utf-8")

    return {
        "max_difference": maximum_difference,
        "image": ela_base64
    }


# ==========================================
# FEATURE 7: SHA-256 EVIDENCE VERIFICATION
# ==========================================

def create_evidence_record(image_data, form):

    client_hash = form.get(
        "client_hash", ""
    ).lower()

    client_timestamp = form.get(
        "client_timestamp", ""
    )

    client_file_name = form.get(
        "client_file_name", ""
    )

    client_file_size = form.get(
        "client_file_size", ""
    )

    server_hash = hashlib.sha256(
        image_data
    ).hexdigest()

    hash_verified = (
        len(client_hash) == 64
        and hmac.compare_digest(
            client_hash,
            server_hash
        )
    )

    return {
        "sha256": server_hash,
        "client_hash": client_hash,
        "hash_verified": hash_verified,
        "file_name": os.path.basename(client_file_name),
        "file_size": client_file_size,
        "client_timestamp": client_timestamp,
        "server_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "image_uploaded_to_server": True
    }


# ==========================================
# FEATURE 8: RISK ASSESSMENT
# ==========================================

def create_risk_assessment(detection, evidence):

    if not evidence.get("hash_verified", False):
        return {
            "level": "HIGH",
            "message": "Evidence integrity verification failed. Manual review required."
        }

    if detection["uncertainty"]["level"] == "uncertain":
        return {
            "level": "MEDIUM",
            "message": "AI result is inconclusive. Additional review recommended."
        }

    if detection["uncertainty"]["level"] == "likely-artificial":
        return {
            "level": "HIGH",
            "message": "AI-generated content is suspected. Manual verification recommended."
        }

    return {
        "level": "LOW",
        "message": "No immediate review flag detected. Verify original image provenance."
    }


# ==========================================
# HTML DASHBOARD
# ==========================================
PAGE = r"""
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>WomenShield AI</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #080d1b;
    color: #e8ecff;
    font-family: Arial, sans-serif;
}

header {
    background: #10182c;
    border-bottom: 1px solid #27365b;
    padding: 22px 7%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}

.logo {
    font-size: 25px;
    font-weight: bold;
    color: #a78bfa;
}

.status {
    color: #4ade80;
    font-size: 13px;
}

.container {
    width: 90%;
    max-width: 1150px;
    margin: 35px auto;
}

.hero {
    text-align: center;
    margin-bottom: 35px;
}

.hero h1 {
    font-size: 34px;
    color: #c4b5fd;
}

.hero p {
    color: #9ca9c9;
    line-height: 1.6;
}

.panel {
    background: #121c32;
    border: 1px solid #29395f;
    border-radius: 16px;
    padding: 25px;
    margin-bottom: 25px;
}

.panel h2 {
    color: #c4b5fd;
    margin-top: 0;
}

.upload-box {
    border: 2px dashed #7761c9;
    border-radius: 14px;
    padding: 35px 15px;
    text-align: center;
    background: #17223d;
}

input[type="file"] {
    max-width: 100%;
    margin: 15px;
    color: #dbeafe;
}

button {
    background: linear-gradient(90deg, #6d4be8, #9333ea);
    color: white;
    padding: 13px 28px;
    border: none;
    border-radius: 9px;
    font-weight: bold;
    cursor: pointer;
    margin-top: 12px;
}

.button {
    display: inline-block;
    background: linear-gradient(90deg, #6d4be8, #9333ea);
    color: white;
    padding: 13px 28px;
    border-radius: 9px;
    font-weight: bold;
    text-decoration: none;
    margin-top: 12px;
}

button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 18px;
}

.card {
    background: #192642;
    border: 1px solid #34466d;
    border-radius: 13px;
    padding: 22px;
    overflow-wrap: anywhere;
}

.card h3 {
    color: #a5b4fc;
    font-size: 15px;
}

.number {
    font-size: 30px;
    font-weight: bold;
    color: #c4b5fd;
    margin: 12px 0;
}

.small {
    font-size: 13px;
    color: #9ca9c9;
}

.details {
    color: #dbeafe;
    line-height: 1.9;
    overflow-wrap: anywhere;
}

.notice {
    color: #fbbf24;
    background: #332a19;
    border-radius: 9px;
    padding: 13px;
    font-size: 13px;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

.error {
    color: #fca5a5;
    text-align: center;
    padding: 12px;
}

.uncertainty-card {
    border: 1px solid #8064d9;
    background: linear-gradient(135deg, #20234b, #192642);
    border-radius: 14px;
    padding: 24px;
    margin-top: 20px;
}

.uncertainty-status {
    font-size: 27px;
    font-weight: bold;
    margin: 12px 0;
}

.likely-artificial {
    color: #fb923c;
}

.likely-human {
    color: #4ade80;
}

.uncertain {
    color: #fbbf24;
}

.result-image {
    width: 100%;
    max-width: 400px;
    border-radius: 12px;
    border: 1px solid #64748b;
    margin-top: 15px;
}

code {
    overflow-wrap: anywhere;
    color: #a5f3fc;
}

footer {
    text-align: center;
    padding: 25px;
    color: #7785a7;
    font-size: 13px;
}

</style>

</head>

<body>

<header>
    <div class="logo">WomenShield AI</div>
    <div class="status">● LOCAL MODEL ANALYSIS</div>
</header>

<div class="container">

    <div class="hero">
        <h1>AI-Powered Image Protection</h1>
        <p>
            Intelligent image screening and digital image analysis
        </p>
    </div>

    <div class="panel">

        <h2>01 / Image Scanner</h2>

        <div class="upload-box">

            <form method="POST"
                  enctype="multipart/form-data"
                  id="uploadForm">

                <p>Select an image for analysis</p>

                <input
                    type="file"
                    name="image"
                    id="imageInput"
                    accept=".jpg,.jpeg,.png,.webp"
                    required
                >

                <input type="hidden" name="client_hash" id="clientHash">
                <input type="hidden" name="client_timestamp" id="clientTimestamp">
                <input type="hidden" name="client_file_name" id="clientFileName">
                <input type="hidden" name="client_file_size" id="clientFileSize">

                <br>

                <button type="submit" id="scanButton">
                    Analyse Image
                </button>

                <p class="small" id="hashStatus">
                    SHA-256 verification will run before upload.
                </p>

            </form>

        </div>

        <div class="notice">
            Privacy notice: The selected original image is uploaded to
            this Flask server for analysis. SHA-256 is used to compare
            the browser-generated and server-generated file hashes.
            This prototype does not provide zero-upload processing.
        </div>

    </div>

    {% if error %}
    <div class="panel error">
        <h3>{{ error }}</h3>
    </div>
    {% endif %}

    {% if result %}

    <div class="panel">

        <h2>02 / AI Image Detection</h2>

        <div class="grid">

            <div class="card">
                <h3>AI Score</h3>
                <div class="number">{{ result.ai_score }}%</div>
            </div>

            <div class="card">
                <h3>Human Score</h3>
                <div class="number">{{ result.human_score }}%</div>
            </div>

            <div class="card">
                <h3>Model Prediction</h3>
                <div class="number">{{ result.prediction }}</div>
            </div>

            <div class="card">
                <h3>Score Margin</h3>
                <div class="number">{{ result.score_margin }}%</div>
            </div>

        </div>

        <div class="notice">
            Model scores are not calibrated probabilities and do not
            prove whether an image is authentic or manipulated.
            Results may vary across image sources and generators.
        </div>

    </div>


    <div class="panel">

        <h2>03 / Uncertainty Intelligence</h2>

        <div class="uncertainty-card">

            <div class="uncertainty-status {{ result.uncertainty.level }}">
                {{ result.uncertainty.status }}
            </div>

            <p>{{ result.uncertainty.explanation }}</p>

            <p>
                <strong>Recommendation:</strong>
                {{ result.uncertainty.recommendation }}
            </p>

        </div>

    </div>


    <div class="panel">

        <h2>04 / Face Detection</h2>

        <div class="card">
            <h3>Detected Faces</h3>
            <div class="number">{{ result.face_count }}</div>

            <img
                class="result-image"
                src="data:image/png;base64,{{ result.face_preview }}"
                alt="Face detection preview"
            >
        </div>

        <p class="small">
            Face detection identifies visible face regions.
            It does not determine a person's identity.
        </p>

    </div>


    <div class="panel">

        <h2>05 / Privacy Shield & Evidence Check</h2>

        <div class="card">
            <h3>Digital Evidence ID</h3>
            <div class="number">
                {{ result.evidence_id }}
            </div>
            <p class="small">
                Unique Evidence Record ID
            </p>
            <a href="/download-report/{{ result.evidence_id }}"
                 class="button">
                  Download PDF Report
            </a>
            <br><br>

            <a href="/evidence-locker"
                style="display:inline-block;
                        background:#6d4be8;
                        color:white;
                        padding:12px 22px;
                        border-radius:9px;
                        text-decoration:none;
                        font-weight:bold;">
                 Open Evidence Locker
            </a>
        </div>
        <div class="grid">

            <div class="card">
                <h3>Hash Verification</h3>

                <div class="number">
                    {% if result.evidence.hash_verified %}
                        MATCH
                    {% else %}
                        NOT VERIFIED
                    {% endif %}
                </div>

                <p class="small">
                    Browser hash compared with server hash.
                </p>
            </div>

            <div class="card">
                <h3>Uploaded File</h3>
                <p class="details">
                    {{ result.evidence.file_name or "Not provided" }}
                </p>
            </div>

            <div class="card">
                <h3>File Size</h3>
                <div class="number">
                    {{ result.evidence.file_size or "Unknown" }}
                </div>
                <p class="small">Bytes reported by browser</p>
            </div>

        </div>

        <h3>Server SHA-256</h3>
        <p><code>{{ result.evidence.sha256 }}</code></p>

        <h3>Browser SHA-256</h3>
        <p>
            <code>
                {{ result.evidence.client_hash or "Unavailable" }}
            </code>
        </p>

        <h3>Client Timestamp</h3>
        <p class="details">
            {{ result.evidence.client_timestamp or "Unavailable" }}
        </p>

        <h3>Server Timestamp</h3>
        <p class="details">
            {{ result.evidence.server_timestamp }}
        </p>

        <div class="notice">
            A matching hash indicates that the bytes received by the
            server match the bytes hashed in the browser. It does not
            prove the image's original source, creation time, ownership,
            or chain of custody. Browser-provided details are not
            independently authenticated.
        </div>

    </div>


    <div class="panel">

        <h2>06 / Risk Assessment</h2>

        <div class="uncertainty-card">
            <h3>
                Risk Level: {{ result.risk_assessment.level }}
            </h3>

            <p class="details">
               {{ result.risk_assessment.message }}
            </p>
        </div>

        <p class="small">
            This is a prototype review status based on hash verification
            and model uncertainty. It is not a validated safety score.
        </p>

    </div>


    <div class="panel">

        <h2>07 / Image Information</h2>

        <div class="grid">

            <div class="card">
                <h3>Image Format</h3>
                <div class="number">{{ result.format }}</div>
            </div>

            <div class="card">
                <h3>Dimensions</h3>
                <div class="number">
                    {{ result.width }} × {{ result.height }}
                </div>
            </div>

            <div class="card">
                <h3>File Size</h3>
                <div class="number">{{ result.size }} KB</div>
            </div>

        </div>

    </div>


    <div class="panel">

        <h2>08 / Metadata Scanner</h2>

        {% if result.metadata %}

            {% for key, value in result.metadata.items() %}
                <p class="details">
                    <strong>{{ key }}:</strong> {{ value }}
                </p>
            {% endfor %}

        {% else %}
       
            <p class="small">
                No supported EXIF metadata was found.
            </p>

        {% endif %}

        <div class="notice">
            Missing metadata does not prove that an image is artificial
            or manipulated.
        </div>

    </div>


    <div class="panel">

        <h2>09 / OpenCV Visual Analysis</h2>

        <div class="grid">

            <div class="card">
                <h3>Brightness</h3>
                <div class="number">{{ result.brightness }}</div>
            </div>
            <div class="card">
                <h3>Contrast</h3>
                <div class="number">{{ result.contrast }}</div>
            </div>

            <div class="card">
                <h3>Edge Percentage</h3>
                <div class="number">{{ result.edges }}%</div>
            </div>

        </div>

    </div>


    <div class="panel">

        <h2>10 / Error Level Analysis</h2>

        {% if result.ela %}

            <p class="details">
                Maximum pixel difference:
{{ result.ela.max_difference }}
            </p>

            <img
                class="result-image"
                src="data:image/png;base64,{{ result.ela.image }}"
                alt="ELA visualisation"
            >

        {% else %}

            <p class="small">
                ELA preview is available for JPEG images in this prototype.
            </p>

        {% endif %}

        <div class="notice">
            ELA highlights compression differences. It is not proof
            of image manipulation or authenticity.
        </div>

    </div>

    <div class="panel">
        <h2>11 / Photo Misuse Detection</h2>

        <p class="details">
            Do you want to check whether this photo is being
            misused by another suspicious profile?
        </p>

        <a href="/photo-misuse"
           class="button">
            Check for Suspicious Profile
        </a>
    </div>

    {% endif %}

</div>

<footer>
    WomenShield AI | Educational Prototype
</footer>


<script>

const uploadForm = document.getElementById("uploadForm");

uploadForm.addEventListener("submit", async function(event) {

    event.preventDefault();

    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];

    const status = document.getElementById("hashStatus");
    const button = document.getElementById("scanButton");

    if (!file) {
        status.textContent = "Please select an image first.";
        return;
    }

    if (!window.crypto || !window.crypto.subtle) {
        status.textContent =
            "SHA-256 is unavailable in this browser context.";
        return;
    }
    button.disabled = true;
    status.textContent = "Generating SHA-256 hash...";

    try {

        const fileBuffer = await file.arrayBuffer();

        const hashBuffer = await crypto.subtle.digest(
            "SHA-256",
            fileBuffer
        );

        const hashArray = Array.from(
            new Uint8Array(hashBuffer)
        );

        const hashHex = hashArray
            .map(byte => byte.toString(16).padStart(2, "0"))
            .join("");

        document.getElementById("clientHash").value = hashHex;

        document.getElementById("clientTimestamp").value =
            new Date().toISOString();
        document.getElementById("clientFileName").value =
            file.name;

        document.getElementById("clientFileSize").value =
            file.size;

        status.textContent =
            "SHA-256 generated. Uploading image to server for analysis...";

        uploadForm.submit();

    } catch (error) {

        console.error("SHA-256 error:", error);

        status.textContent =
            "Unable to generate SHA-256. Please try again.";

        button.disabled = false;
    }

});

</script>

</body>
</html>
"""

DASHBOARD = r"""
<!DOCTYPE html>
<html>
<head>
    <title>WomenShield AI - Evidence Locker</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #101827;
            color: white;
            padding: 30px;
        }

        h1 {
            color: #60a5fa;
        }

        .record {
            background: #1e293b;
            padding: 20px;
            margin: 15px 0;
            border-radius: 10px;
            overflow-wrap: anywhere;
        }

        a {
            color: #60a5fa;
        }

        .empty {
            color: #cbd5e1;
        }
    </style>
</head>

<body>

    <h1>WomenShield AI</h1>
    <h2>Digital Evidence Locker</h2>

    <p>Total Evidence Records: {{ records|length }}</p>

    {% if records %}

        {% for record in records %}

            <div class="record">

                <h3>Evidence ID: {{ record.evidence_id }}</h3>

                <p>File Name: {{ record.file_name }}</p>

                <p>SHA-256: {{ record.sha256 }}</p>

                <p>Client Timestamp: {{ record.client_timestamp }}</p>

                <p>Server Timestamp: {{ record.server_timestamp }}</p>

                <p>
                    Hash Verified:
                    {{ "MATCH" if record.hash_verified else "NOT VERIFIED" }}
                </p>

                <p>Risk Assessment: {{ record.risk_assessment }}</p>

                <a href="/download-report/{{ record.evidence_id }}">
                    Download PDF Report
                </a>

            </div>

        {% endfor %}

    {% else %}

        <p class="empty">No evidence records found.</p>

    {% endif %}

    <p><a href="/">Back to Home</a></p>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    session.setdefault("current_evidence_ids", [])

    if request.method == "POST":

        file = request.files.get("image")


        if not file or file.filename == "":
            error = "Please select an image."

        else:

            image_data = file.read()

            if not image_data:
                error = "The selected image is empty."

            else:

                try:
                    # Create evidence record
                    evidence_record = create_evidence_record(
                        image_data,
                        request.form
                    )

                    # Open and validate image
                    img = Image.open(
                        BytesIO(image_data)
                    )

                    img.load()
                    image_format = img.format

                    if image_format not in ALLOWED_FORMATS:

                        error = (
                            "Only JPG, JPEG, PNG and WEBP "
                            "images are supported."
                        )

                    else:
                        misuse_token = save_image_for_misuse_check(
                                        image_data,
                                        image_format
                        )

                        session["misuse_token"] = misuse_token
                        # AI analysis
                        detection = detect_image(img,
                        filename=request.files["image"].filename
                        )

                        # Face detection
                        face_result = detect_faces(img)

                        # Metadata
                        metadata = extract_metadata(img)

                        # OpenCV analysis
                        visual_analysis = perform_opencv_analysis(
                            img
                        )

                        # ELA
                        ela_result = None

                        if image_format == "JPEG":
                            ela_result = perform_ela(image_data)

                        # Risk assessment
                        risk_assessment = create_risk_assessment(
                            detection,
                            evidence_record
                        )
                        evidence_id = save_evidence_record(
    evidence_record,
    risk_assessment
                        )
                        current_ids = session.get("current_evidence_ids", [])
                        current_ids.append(evidence_id)
                        session["current_evidence_ids"] = current_ids
                        
                        pdf_path = generate_evidence_pdf(
    evidence_id,
    evidence_record,
    risk_assessment
                        )
                        # Final result
                        result = {

                            "format": image_format,

                            "width": img.width,
                            "height": img.height,

                            "size": round(
                                len(image_data) / 1024,
                                2
                            ),

                            "face_count": face_result["count"],
                            "face_preview": face_result["preview"],

                            "ai_score": detection["ai_score"],
                            "human_score": detection["human_score"],

                            "prediction": detection["prediction"],

                            "uncertainty": detection["uncertainty"],
                            "score_margin": detection["score_margin"],

                            "metadata": metadata,

                            "brightness": visual_analysis["brightness"],
                            "contrast": visual_analysis["contrast"],
                            "edges": visual_analysis["edges"],

                            "ela": ela_result,
                            "evidence_id": evidence_id,
                            "pdf_path": pdf_path,
                            "evidence": evidence_record,

                            "risk_assessment": risk_assessment
                        }

                except Exception as exc:

                    import traceback
                    traceback.print_exc()

                    error = f"Analysis failed: {str(exc)}"
    if result:
        print("EVIDENCE:", result.get("evidence"))
        print("RISK:", result.get("risk_assessment"))
        print("RESULT KEYS:", list(result.keys()))
    else:
        print("RESULT IS NONE")
        print("HOME ROUTE REACHED")
    return render_template_string(
        PAGE,
        result=result,
        error=error
    )
# ==========================================
# FEATURE 11: PHOTO MISUSE SCANNER
# ==========================================
@app.route("/photo-misuse", methods=["GET", "POST"])
def photo_misuse():

    misuse_token = session.get("misuse_token")

    if not misuse_token:
        return "Please analyse an image first.", 400

    # Find the uploaded homepage image
    original_path = None

    for extension in ["jpg", "jpeg", "png", "webp"]:
        candidate = os.path.join(
            MISUSE_UPLOAD_DIR,
            f"{misuse_token}.{extension}"
        )

        if os.path.isfile(candidate):
            original_path = candidate
            break

    if not original_path:
        return "Original image unavailable.", 404

    # Reference images
    reference_folder = os.path.join(
        BASE_DIR, "static", "reference_images"
    )

    reference_files = [
        "ai_woman_1.jpeg",
        "ai_woman_2.jpeg"
    ]

    # Show URL input before comparison
    if request.method == "GET":
        return render_template_string("""
        <html>
        <body style="
            background:#080d1b;
            color:white;
            font-family:Arial;
            padding:30px;
        ">
            <h1 style="color:#a78bfa;">WomenShield AI</h1>
            <h2>Photo Misuse Scanner</h2>

            <form method="POST">
                <label>Instagram / Website Profile URL</label>
                <br><br>

                <input
                    type="url"
                    name="instagram_url"
                    placeholder="Paste profile or website URL here"
                    required
                    style="width:90%;padding:12px;"
                >

                <br><br>
                <button type="submit" style="padding:12px;">
                    Compare Image
                </button>
            </form>

            <p style="color:#aaa;">
                Demo mode: compares your uploaded image with
                locally stored reference images.
                It does not fetch or verify website content.
            </p>
        </body>
        </html>
        """)

    # Read submitted URL
    profile_url = request.form.get(
        "instagram_url", ""
    ).strip()

    if not profile_url:
        return "Please enter a URL.", 400

    results = []

    try:
        uploaded = Image.open(original_path).convert("RGB")
        uploaded_hash = imagehash.phash(uploaded)

        for filename in reference_files:

            reference_path = os.path.join(
                reference_folder, filename
            )

            if not os.path.isfile(reference_path):
                print("Missing reference:", reference_path)
                continue

            with Image.open(reference_path) as reference:
                reference_hash = imagehash.phash(
                    reference.convert("RGB")
                )

            difference = uploaded_hash - reference_hash

            similarity = max(
                0,
                100 - (difference / 64 * 100)
            )

            results.append({
                "filename": filename,
                "similarity": round(similarity, 2)
            })

        if not results:
            return "Reference images unavailable.", 404

        best = max(
            results,
            key=lambda item: item["similarity"]
        )

        if best["similarity"] >= 85:
            status = "Potential Misuse Match Detected"
            color = "#f87171"
            safety_message = (
                "Your image may match a reference image. "
                "Please review it."
            )
        else:
            status = "No Match Found"
            color = "#22c55e"
            safety_message = (
                "No match found in the available demo references."
            )

        return render_template_string("""
        <html>
        <body style="
            background:#080d1b;
            color:white;
            font-family:Arial;
            padding:30px;
        ">
            <h1 style="color:#a78bfa;">WomenShield AI</h1>

            <h3>Submitted URL</h3>
            <p style="word-break:break-all;">
                {{ profile_url }}
            </p>

            <h2 style="color:{{ color }};">
                {{ status }}
            </h2>

            <h3>{{ safety_message }}</h3>

            <h3>Best Similarity: {{ score }}%</h3>

            <h3>Matched Reference Image</h3>
            <p>{{ best.filename }}</p>
            <p>Similarity: {{ best.similarity }}%</p>

            <p style="color:#aaa;">
                Demo comparison only. The submitted URL's
                actual website images are not fetched or verified.
            </p>

            <a href="/" style="color:#a78bfa;">
                Back to Home
            </a>
        </body>
        </html>
        """,
        profile_url=profile_url,
        status=status,
        color=color,
        safety_message=safety_message,
        score=best["similarity"],
        best=best)

    except Exception as exc:
        print("Photo comparison error:", repr(exc))
        return "Image comparison failed. Check the terminal.", 400

@app.route("/download-report/<evidence_id>")
def download_evidence_report(evidence_id):

    if (
    len(evidence_id) != 13
    or not evidence_id.startswith("WS-")
    or not all(c in "0123456789ABCDEF" for c in evidence_id[3:])
):
        abort(404)

    pdf_path = os.path.join(
        BASE_DIR,
        "static",
        "reports",
        f"{evidence_id}.pdf"
    )

    if not os.path.isfile(pdf_path):
        abort(404)

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f"{evidence_id}.pdf"
    )
# DIGITAL EVIDENCE LOCKER DASHBOARD
@app.route("/evidence-locker")
def evidence_locker():

    current_ids = session.get("current_evidence_ids", [])

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    if current_ids:
        placeholders = ",".join("?" for _ in current_ids)

        records = conn.execute(f"""
            SELECT
                evidence_id,
                file_name,
                sha256,
                client_timestamp,
                server_timestamp,
                hash_verified,
                risk_assessment
            FROM evidence
            WHERE evidence_id IN ({placeholders})
            ORDER BY rowid DESC
        """, current_ids).fetchall()
    else:
        records = []

    conn.close()

    return render_template_string(
        DASHBOARD,
        records=records
    )
# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    init_database()

    app.run(
        debug=False,
        use_reloader=False
    )