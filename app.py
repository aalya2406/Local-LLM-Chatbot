import json
from flask import Flask, request, jsonify, send_from_directory
import pdfplumber
import os
from langchain_community.llms import Ollama  # Assuming Ollama is available through langchain_community

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# Initialize Ollama
llm = Ollama(model="llama2")

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"Page {page_num + 1}:\n{page_text}\n"
                else:
                    print(f"Warning: No text found on page {page_num + 1}")
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

@app.route('/analyze', methods=['POST'])
def analyze_pdf():
    file = request.files['file']
    question = request.form['question']
    pdf_path = f"/tmp/{file.filename}"
    file.save(pdf_path)
    
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Print extracted text for debugging
    print("Extracted Text:")
    print(text)
    
    # Feed the document text to Ollama and ask the question
    ollama_input = f"Document:\n{text}\n\nQuestion: {question}"
    
    # Invoke Ollama with the input text
    ollama_response = llm.invoke(ollama_input)
    
    # Extract the relevant part of the response
    try:
        ollama_response_json = json.loads(ollama_response)
        answer = ollama_response_json.get('response', "No relevant answer found")
    except json.JSONDecodeError:
        answer = ollama_response or "No relevant answer found"
    
    return jsonify({"answer": answer})

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
