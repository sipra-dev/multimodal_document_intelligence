📄 Multimodal Document Intelligence System


    An AI-powered multimodal document processing system that combines OCR, Vision Language Models, Machine Learning classification, and structured extraction to intelligently analyze business documents.

The system can automatically process and understand:

    Supplier Invoices
    
    Sales Receipts
    
    Inventory Reports
    
    Analytics Reports


Built using OpenAI Vision models, OCR pipelines, Scikit-learn classification, Pydantic validation, and Streamlit.

🚀 Features

    📄 Upload PDF business documents

    🧠 Multimodal document understanding using:

            OCR extraction
            Vision AI analysis
            Statistical NLP classification

    📑 Automatic document classification:

            Supplier Invoice
            Sales Receipt
            Inventory Report
            Analytics Report

    ⚡ Structured JSON extraction using Pydantic schemas

    👨‍💻 Human-in-the-Loop validation for low-confidence predictions
    
    📊 Automatic extraction of:

            Invoice details
            Receipt information
            Inventory summaries
            Business analytics metrics

    💾 Clean JSON output generation and storage

    🖼️ Handles scanned PDFs and image-heavy documents

🧠 Tech Stack

    Python
    
    Streamlit (Frontend)
    
    OpenAI GPT-4.1-mini Vision
    
    PyMuPDF (fitz)
    
    Tesseract OCR
    
    Scikit-learn
    
    TF-IDF Vectorizer
    
    Logistic Regression
    
    Pydantic
    
    Joblib
    
    Pillow

🏗️ Architecture

    PDF Upload → PDF to Images → OCR Extraction → Vision AI Analysis → ML Classification → Confidence Check → HITL Validation → Structured Extraction → JSON Output

📌 How It Works

    User uploads a PDF document
    
    PDF pages are converted into images
    
    OCR extracts raw textual information
    
    Vision AI analyzes layouts, tables, and visual structure
    
    OCR + Vision outputs are combined
    
    ML classifier predicts document type
    
    If confidence is low, Human-in-the-Loop verification is triggered
    
    Structured extraction generates validated JSON outputs
    
    Final processed output is saved automatically

📂 Supported Document Types
    Supplier Invoice

        Extracts:
        
        Invoice number
        Supplier name
        Tax
        Discount
        Total amount
        Item list
        
       
    Sales Receipt

        Extracts:
        
        Receipt ID
        Customer details
        Purchase summary
        Taxes
        Product information
        
    Inventory Report

        Extracts:
        
        Product inventory
        Stock summaries
        Inventory statistics

    Analytics Report

        Extracts:
        
        Metrics
        Business insights
        Analytical summaries

        
⚡ Human-in-the-Loop (HITL)

If the classifier confidence falls below the threshold:

the system pauses automatic processing
asks the user to verify the document category
reduces incorrect downstream extraction

This improves reliability in real-world document processing scenarios.

