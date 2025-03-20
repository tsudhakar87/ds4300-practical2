# This file is in charge of extracting and preprocessing the text
# from the PDF files of the class materials.

import os
import string
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader

class Preprocessor:
    def __init__(self, data_dir, chunk_size=300, overlap=50, text_prep='all'):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_prep = text_prep.lower()
        self.stop_words = set(stopwords.words("english"))

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF using PyPDF2.
        """
        reader = PdfReader(pdf_path)
        return [(i + 1, page.extract_text()) for i, page in enumerate(reader.pages) if page.extract_text()]

    def clean_text(self, text):
        """
        Cleans text based on the specified text preprocessing strategy.
        """
        text = text.lower().strip()

        if self.text_prep in ['punctuation removal', 'all']:
            text = text.translate(str.maketrans('', '', string.punctuation))

        if self.text_prep in ['whitespace removal', 'all']:
            text = " ".join(text.split())

        if self.text_prep == 'all':
            text = text.replace("●", "")
            words = text.split()
            text = " ".join([word for word in words if word not in self.stop_words])
        
        return text

    def split_text_into_chunks(self, text):
        """
        Splits cleaned text into overlapping chunks.
        """
        words = text.split()
        chunks = [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size - self.overlap)]
        return chunks

    def process_pdfs(self):
        """
        Processes all PDFs in the directory: extract, clean, and chunk text.
        Returns a list of (file_name, page_num, chunk_index, chunk) tuples.
        """
        all_chunks = []
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(self.data_dir, file_name)
                text_by_page = self.extract_text_from_pdf(pdf_path)
                print(f"Processing {file_name}")

                for page_num, text in text_by_page:
                    cleaned_text = self.clean_text(text)
                    chunks = self.split_text_into_chunks(cleaned_text)

                    for chunk_index, chunk in enumerate(chunks):
                        all_chunks.append((file_name, page_num, chunk_index, chunk))

        return all_chunks
