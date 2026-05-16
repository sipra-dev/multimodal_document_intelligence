import streamlit as st
import tempfile
import json

from multimodal_backend import process_document



# PAGE CONFIG


st.set_page_config(
    page_title="Multimodal Document Intelligence",
    layout="wide"
)

st.title("📄 Multimodal Document Intelligence System")



# INPUT

uploaded_pdf = st.file_uploader(
    "Upload PDF Document",
    type=["pdf"]
)

csv_path = st.text_input(
    "Dataset CSV Path",
    value=r"C:\Users\Sipra Neye\document_dataset1.csv"
)



# PROCESS BUTTON


if "temp_pdf_path" not in st.session_state:
    st.session_state.temp_pdf_path = None

if "review_required" not in st.session_state:
    st.session_state.review_required = False

if "backend_response" not in st.session_state:
    st.session_state.backend_response = None



if st.button("Process Document"):

    if uploaded_pdf is None:

        st.error("Please upload a PDF file.")

    else:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_pdf.read())

            st.session_state.temp_pdf_path = tmp.name

        output = process_document(
            st.session_state.temp_pdf_path,
            csv_path
        )

        
        # LOW CONFIDENCE
        

        if output["needs_review"]:

            st.session_state.review_required = True

            st.session_state.backend_response = output

        else:

            st.success("Processing Complete!")

            st.json(output["result"])



# HUMAN IN THE LOOP UI


if st.session_state.review_required:

    resp = st.session_state.backend_response

    st.warning(
        f"""
        Your document is predicted as:

        {resp['predicted_type']}

        But I am not confident enough.

        Confidence Score: {resp['confidence']:.2f}

        Please choose the correct document type.
        """
    )

    choice = st.radio(
        "Choose document type:",
        [
            "supplier_invoice",
            "sales_receipt",
            "inventory_report",
            "analytics_report"
        ]
    )

    if st.button("Submit Choice"):

        final_output = process_document(
            st.session_state.temp_pdf_path,
            csv_path,
            user_choice=choice
        )

        st.success("Processing Complete!")

        st.subheader("Final Output")

        st.json(final_output["result"])

        st.session_state.review_required = False