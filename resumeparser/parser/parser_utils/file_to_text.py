import pdfplumber
import docx2txt
from io import BytesIO

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            text = "".join([page.extract_text() or "" for page in pdf.pages])
    elif uploaded_file.name.endswith(".docx"):
        temp_file = BytesIO(uploaded_file.read())
        text = docx2txt.process(temp_file)
    else:
        raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")
    return text
