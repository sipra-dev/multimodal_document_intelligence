📄 Multimodal Document Intelligence System

A full-stack AI-powered system that automatically classifies documents, extracts structured information, and supports Human-in-the-Loop (HITL) validation for uncertain predictions.

It combines OCR, Vision LLM extraction, and a machine learning classifier to intelligently process PDFs like invoices, receipts, and reports.

🚀 Features
      
      📄 PDF to Image Conversion using PyMuPDF
      
      🔍 OCR Extraction using Tesseract
      
      🧠 Vision-based Understanding using GPT-4.1-mini
      
      🤖 ML Document Classifier (TF-IDF + Logistic Regression)
      
      🧾 Structured JSON Extraction using Pydantic schemas
      
      ⚖️ Confidence-based HITL System
      
      🧑‍💻 Streamlit UI for easy interaction
      
      💾 Caching system for faster repeated processing
      
      📊 Supports multiple document types:
      
                  Supplier Invoice
                  
                  Sales Receipt
                  
                  Inventory Report
                  
                  Analytics Report
                  
🏗️ System Architecture

          PDF Input
          
          Convert PDF → Images (PyMuPDF)
          
          Run:
          
          OCR (Tesseract)
          
          Vision LLM extraction (GPT-4.1-mini)
          
          Combine extracted text
          
          ML classifier predicts document type

          If confidence is low → Human validation (HITL)
          
          Structured extraction via LLM + Pydantic schema validation
          
          Final processed output saved as JSON
          
📦 Tech Stack

          Python 🐍
          
          Streamlit 🎈

          OpenAI API (GPT-4.1-mini)
          
          PyMuPDF (fitz)
          
          Tesseract OCR
          
          Scikit-learn (TF-IDF + Logistic Regression)
          
          Pydantic
          
          Pandas

🧠 How It Works

          1. Document Classification
          
          A TF-IDF + Logistic Regression model predicts document type.
          
          If:
          
          confidence < 0.85
          
          → User is asked to manually confirm the document type.
          
          2. Multimodal Extraction
          
          Each PDF page is processed using:
          
          OCR (text extraction)
          Vision LLM (semantic understanding)
          
          Both outputs are combined for better accuracy.
          
          3. Structured Output
          
          Data is extracted into strict schemas using Pydantic:
          
          SupplierInvoice
          SalesReceipt
          InventoryReport
          AnalyticsReport
          
          Ensures consistent JSON output.
          
          4. Human-in-the-Loop (HITL)
          
          If model confidence is low:
          
          System shows predicted type
          User selects correct type
          Pipeline re-runs with corrected label
          📁 Output Format
          
          Final output is saved in:
          
          outputs/output.json
          
          Example structure:
          
          {
            "type": "sales_receipt",
            "confidence": 0.91,
            "structured": {
              "receipt_id": "12345",
              "total_amount": 999.0,
              "items": []
            },
            "pipeline": {
              "status": "processed",
              "data": {}
            }
          }
⚡ Key Design Highlights

          🔁 Hybrid AI pipeline (OCR + Vision + ML)
          
          🧠 Confidence-based decision system
          
          🧾 Schema-driven structured extraction
          
          ⚡ Cached OCR + Vision calls for performance
          
          🧑 Human fallback for reliability
          
          📦 Modular backend design (easy to extend)
          
