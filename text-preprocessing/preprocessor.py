from PyPDF2 import PdfReader


# make function 
reader = PdfReader('class-materials/slides/01 - Introduction & Getting Started.pdf')

pages = reader.pages
documents = []

for page in pages: 
    documents.append(page.extract_text())

print(documents)   
