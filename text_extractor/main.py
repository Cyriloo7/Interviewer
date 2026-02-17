import streamlit as st 
import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import pymupdf
import zipfile
from docx import Document
from typing import TypedDict, Annotated
import pandas as pd
import shutil
from docx import Document  # ✅ add for DOCX support

st.set_page_config(page_title="AI Resume Parser", page_icon="📄", layout="wide")

# --- CUSTOM CSS & STYLING ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styling */
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2c3e50;
        text-align: center;
        padding: 20px;
        font-weight: 800;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Card-like container for the uploader */
    .upload-card {
        background: rgba(255, 255, 255, 0.7);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 25px;
    }
    
    /* Button Styling */
    div.stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 15px;
        font-size: 18px;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    div.stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- ENV ---
load_dotenv()
key = os.getenv("gemini")
if not key:
    st.error("Missing API key. Add `gemini=YOUR_KEY` to your .env file.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = key

st.markdown('<h1 class="main-header">📄 Resume Intelligence Extractor</h1>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    # ✅ allow zip + pdf (keeps your CSS/layout unchanged)
    uploaded_file = st.file_uploader(
        "Drop your ZIP file containing resumes here (or upload a single PDF)",
        type=["zip", "pdf"]
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    submit_clicked = st.button("Process Resumes")

class ExtractText(TypedDict):
    name: Annotated[str, "Extract the Name of the candidate from the resume"]
    summary: Annotated[str, "Extract only the Summary of the resume provided not extract the phone number or email or github links"]
    exp: Annotated[int, "Extract the experience mentioned in the resume if present else return 0"]
    skills: Annotated[list[str], "Extract the skills from the resume"]
    links: Annotated[list[str], "Extract the links from the resume as list of strings"]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
llm = model.with_structured_output(ExtractText)

EXTRACT_DIR = "extracted_pdfs"

def extract_text(path: str) -> str:
    """Extract text from PDF or DOCX."""
    lower = path.lower()
    if lower.endswith(".pdf"):
        doc = pymupdf.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    if lower.endswith(".docx"):
        d = Document(path)
        return "\n".join(p.text for p in d.paragraphs if p.text.strip())

    return ""

if submit_clicked:
    if uploaded_file:
        with st.status("Analyzing Resumes...", expanded=True) as status:
            extracted_data = []

            # reset output dir
            if os.path.exists(EXTRACT_DIR):
                shutil.rmtree(EXTRACT_DIR)
            os.makedirs(EXTRACT_DIR, exist_ok=True)

            file_name_lower = uploaded_file.name.lower()

            # ✅ CASE 1: user uploads a single PDF
            if file_name_lower.endswith(".pdf"):
                save_path = os.path.join(EXTRACT_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                files_list = [uploaded_file.name]

            # ✅ CASE 2: user uploads a ZIP of resumes
            else:
                with zipfile.ZipFile(uploaded_file, 'r') as zip_f:
                    zip_f.extractall(EXTRACT_DIR)

                files_list = [
                    f for f in os.listdir(EXTRACT_DIR)
                    if f.lower().endswith(".pdf") or f.lower().endswith(".docx")
                ]

            # process files
            for file in files_list:
                st.write(f"Processing: `{file}`")
                path = os.path.join(EXTRACT_DIR, file)

                text = extract_text(path)
                if not text.strip():
                    st.warning(f"Skipped `{file}` (no text extracted).")
                    continue

                response = llm.invoke(text)

                extracted_info = {
                    "Name": response.get('name', ''),
                    "Summary": response.get('summary', ''),
                    "Experience (Yrs)": response.get('exp', 0),
                    "Skills": ", ".join(response.get('skills', [])),
                    "Links": ", ".join(response.get('links', []))
                }
                extracted_data.append(extracted_info)

            status.update(label="Extraction Complete!", state="complete", expanded=False)

        if extracted_data:
            df = pd.DataFrame(extracted_data)

            st.download_button(
                label="📥 Download Results as CSV",
                data=df.to_csv(index=False),
                file_name="extracted_resumes.csv",
                mime="text/csv"
            )
        else:
            st.error("No resumes were processed. Make sure the ZIP contains readable PDFs/DOCX or upload a valid PDF.")
    else:
        st.error("Please upload a ZIP file or a PDF first.")
