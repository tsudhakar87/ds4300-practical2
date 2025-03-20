# This file is in charge of extracting and preprocessing the text
# from the PDF files of the class materials.

import os
import string
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader

stop_words = set(stopwords.words("english"))

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = reader.pages
    text_list = [page.extract_text() for page in pages if page.extract_text()]
    return text_list

def clean_text(text_list):
    cleaned_text = []

    for text in text_list:
        text = text.lower().strip() 
        text = text.translate(str.maketrans('', '', string.punctuation)) 
        text = text.replace("●", "")
        words = text.split()
        cleaned_text.append(" ".join([word for word in words if word not in stop_words])) 
    
    return cleaned_text

def process_pdfs_in_folder(folder_path, output_filename):
    all_cleaned_text = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"): 
            pdf_path = os.path.join(folder_path, filename)
            extracted_text_list = extract_text_from_pdf(pdf_path) 
            cleaned_text_list = clean_text(extracted_text_list)  

            
            all_cleaned_text.extend(cleaned_text_list)    

    final_text = "\n".join(all_cleaned_text)  

    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(final_text) 

    print("Processed text saved to:", output_filename)
    return final_text  

 
folder_path = "class-materials/slides" 
output_file = "cleaned_text.txt"

cleaned_text = process_pdfs_in_folder(folder_path, output_file)
