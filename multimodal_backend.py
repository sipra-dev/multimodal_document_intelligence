import os
import json
import base64
import fitz
import pytesseract
import joblib
import pandas as pd
import hashlib

from PIL import Image
from openai import OpenAI
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


# CONFIG


CONFIDENCE_THRESHOLD = 0.85
OUTPUT_DIR = "outputs"
MODEL_PATH = "model.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# PDF -> IMAGE


def pdf_to_images(pdf_path, output_dir="temp_pages"):
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)

    images = []

    for i in range(len(doc)):
        page = doc.load_page(i)

        pix = page.get_pixmap()

        path = f"{output_dir}/page_{i}.png"

        pix.save(path)

        images.append(path)

    return images



# OCR


def run_ocr(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)



# VISION


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def vision_extract(path):

    img_b64 = encode_image(path)

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a document intelligence system."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all information from this document image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        temperature=0.0
    )

    return res.choices[0].message.content



# CACHE


@lru_cache(maxsize=256)
def cached_ocr(image_path):
    return run_ocr(image_path)


VISION_CACHE = {}

def cached_vision(image_path):

    if image_path in VISION_CACHE:
        return VISION_CACHE[image_path]

    out = vision_extract(image_path)

    VISION_CACHE[image_path] = out

    return out


def make_key(text):
    return hashlib.md5(text.encode()).hexdigest()


PRED_CACHE = {}

def cached_predict(model_obj, text):

    key = make_key(text)

    if key in PRED_CACHE:
        return PRED_CACHE[key]

    pred = model_obj.predict(text)

    PRED_CACHE[key] = pred

    return pred



# CLASSIFIER


class StatisticalDocumentClassifier:

    def __init__(self):

        self.model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2)
                )
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            )
        ])

    def train_from_csv(self, csv_path):

        df = pd.read_csv(csv_path).dropna()

        x = df.iloc[:, 0].astype(str).tolist()
        y = df.iloc[:, 1].astype(str).tolist()

        self.model.fit(x, y)

    def predict(self, text):

        probs = self.model.predict_proba([text])[0]

        pred = self.model.predict([text])[0]

        return pred, float(max(probs))

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)



# LOAD / TRAIN MODEL


def get_model(csv_path):

    clf = StatisticalDocumentClassifier()

    if os.path.exists(MODEL_PATH):

        print("Loading existing model...")

        clf.load(MODEL_PATH)

    else:

        print("Training model for first time...")

        clf.train_from_csv(csv_path)

        clf.save(MODEL_PATH)

        print("Model saved!")

    return clf




# PYDANTIC SCHEMAS


class SupplierInvoice(BaseModel):

    invoice_number: Optional[str] = ""
    supplier_name: Optional[str] = ""

    subtotal: float = 0.0
    tax: float = 0.0
    discount: float = 0.0
    total_amount: float = 0.0

    items: List[Any] = []


class SalesReceipt(BaseModel):

    receipt_id: Optional[str] = ""
    customer_name: Optional[str] = ""

    subtotal: float = 0.0
    tax: float = 0.0
    discount: float = 0.0
    total_amount: float = 0.0

    items: List[Any] = []


class InventoryReport(BaseModel):

    products: List[Any] = []
    stock_summary: Dict[str, Any] = {}


class AnalyticsReport(BaseModel):

    metrics: Dict[str, Any] = {}
    insights: List[str] = []


SCHEMA_MAP = {
    "supplier_invoice": SupplierInvoice,
    "sales_receipt": SalesReceipt,
    "inventory_report": InventoryReport,
    "analytics_report": AnalyticsReport
}



# STRUCTURED EXTRACTION


def structured_extraction(full_text, doc_type):

    schema = SCHEMA_MAP[doc_type]

    prompt = f"""
OCR:
{full_text["ocr"]}

VISION:
{full_text["vision"]}

Return ONLY valid JSON matching schema:
{schema.model_json_schema()}
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Extract structured JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0
    )

    try:

        data = json.loads(res.choices[0].message.content)

        schema.parse_obj(data)

        return data

    except:

        return {
            "raw": res.choices[0].message.content
        }



# PIPELINES


def pipeline(data):

    return {
        "status": "processed",
        "data": data
    }


PIPELINES = {
    "supplier_invoice": pipeline,
    "sales_receipt": pipeline,
    "inventory_report": pipeline,
    "analytics_report": pipeline
}



# SAVE


def clean(obj):

    import numpy as np

    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean(i) for i in obj]

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    return obj


def save(data):

    path = os.path.join(OUTPUT_DIR, "output.json")

    safe_data = clean(data)

    with open(path, "w") as f:
        json.dump(safe_data, f, indent=2)

    return path



# MAIN PIPELINE


def process_document(
    pdf_path,
    csv_path,
    user_choice=None
):

    clf = get_model(csv_path)

    images = pdf_to_images(pdf_path)

    ocr_text = ""
    vision_text = ""

    for img in images:

        ocr_text += cached_ocr(img) + "\n"

        vision_text += cached_vision(img) + "\n"

    full_text = {
        "ocr": ocr_text,
        "vision": vision_text
    }

    pred, conf = cached_predict(
        clf,
        ocr_text + vision_text
    )

    
    # HITL CHECK
    

    if conf < CONFIDENCE_THRESHOLD and user_choice is None:

        return {
            "needs_review": True,
            "predicted_type": pred,
            "confidence": conf
        }

    if user_choice is not None:
        pred = user_choice

    
    # STRUCTURED EXTRACTION
    

    structured = structured_extraction(full_text, pred)

    if "raw" in structured:
        raise Exception("Invalid structured output")

    result = PIPELINES[pred](structured)

    final = {
        "type": pred,
        "confidence": conf,
        "structured": structured,
        "pipeline": result
    }

    path = save(final)

    return {
        "needs_review": False,
        "saved_path": path,
        "result": final
    }