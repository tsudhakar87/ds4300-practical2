# This file is in charge of extracting and preprocessing the text
# from the PDF files of the class materials.

import os
import string
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader

stop_words = set(stopwords.words("english"))

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a pdf path.
    """
    reader = PdfReader(pdf_path)
    text_list = [(i+1, page.extract_text()) for i, page in enumerate(reader.pages) if page.extract_text()]
    return text_list

def clean_text(text):

    text = text.lower().strip() 
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = text.replace("●", "")
    words = text.split()
    cleaned_text = " ".join([word for word in words if word not in stop_words])
    
    return cleaned_text

def split_text_into_chunks(text, chunk_size=300, overlap=50):

    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def process_pdfs(data_dir):

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            
            for page_num, text in text_by_page:
                cleaned_text = clean_text(text)
                chunks = split_text_into_chunks(cleaned_text)
                print(f"  Chunks: {chunks}")

            '''
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")'
            '''

process_pdfs('class-materials/slides')