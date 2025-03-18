from PyPDF2 import PdfReader
from nltk.corpus import stopwords
import nltk
import string

# make function 
reader = PdfReader('class-materials/slides/01 - Introduction & Getting Started.pdf')

pages = reader.pages
documents = []

for page in pages: 
    documents.append(page.extract_text())

print(documents)   



nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def clean_text(text_list):
    cleaned_text = []

    for text in text_list:
        text = text.lower()  
        text = text.strip()  
        text = text.translate(str.maketrans('', '', string.punctuation))


        words = text.split()
        text = " ".join([word for word in words if word not in stop_words])

        cleaned_text.append(text)
    
    return cleaned_text 