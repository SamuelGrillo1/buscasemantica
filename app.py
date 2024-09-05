import os
import nltk
import streamlit as st
import numpy as np
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from io import BytesIO
from PIL import Image

nltk.download('punkt')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def load_and_preprocess_documents(doc_directory):
    documents = []
    for filename in os.listdir(doc_directory):
        if filename.endswith(".txt") or filename.endswith(".pdf") or filename.endswith(".docx"):
            filepath = os.path.join(doc_directory, filename)
            text, pages = extract_text_from_file(filepath)  
            if text:
           
                tokens = nltk.word_tokenize(text)
                documents.append({
                    'filename': filename,
                    'content': " ".join(tokens),
                    'filepath': filepath,  
                    'pages': pages  
                })
    return documents


def extract_text_from_file(filepath):
    text = ""
    pages = []
    if filepath.endswith(".txt"):
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
    elif filepath.endswith(".pdf"):
        text, pages = extract_text_from_pdf(filepath)
    return text, pages


def extract_text_from_pdf(filepath):
    text = ""
    pages = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    pages.append(page_num + 1)  
    except Exception as e:
        st.error(f"Erro ao ler o PDF: {e}")
    return text, pages


def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  
    return embeddings


def calculate_similarity(query_embedding, doc_embeddings):
    cosine_similarities = cosine_similarity(query_embedding.detach().numpy(), doc_embeddings.detach().numpy())
    return cosine_similarities.flatten()


def search_best_practices(query, documents):
    query_embedding = get_bert_embeddings(query)
    doc_embeddings = torch.vstack([get_bert_embeddings(doc['content']) for doc in documents])
    
    cosine_similarities = calculate_similarity(query_embedding, doc_embeddings)
    related_docs_indices = cosine_similarities.argsort()[::-1]
    
    results = []
    for idx in related_docs_indices[:5]:  
        doc = documents[idx]
        results.append((doc['filename'], doc['content'], doc['filepath'], doc['pages'], cosine_similarities[idx]))
    return results


def extract_relevant_snippets(query, document, pages, top_n=3):
    sentences = nltk.sent_tokenize(document)
    query_embedding = get_bert_embeddings(query)
    sentence_embeddings = torch.vstack([get_bert_embeddings(sentence) for sentence in sentences])
    
    similarities = calculate_similarity(query_embedding, sentence_embeddings)
    top_indices = similarities.argsort()[::-1][:top_n]
    
    snippets = [(sentences[i], pages[i]) for i in top_indices]  


def get_pdf_page_image(filepath, page_number):
    try:
        with pdfplumber.open(filepath) as pdf:
            page = pdf.pages[page_number - 1]
            image = page.to_image(resolution=200)

            img_buffer = BytesIO()
            image.original.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            return img_buffer
    except Exception as e:
        st.error(f"Erro ao obter a imagem da página: {e}")
        return None


base_dir = os.path.dirname(__file__)


doc_directory = os.path.join(base_dir, "Dataset_BS")


documents = load_and_preprocess_documents(doc_directory)


st.title("Ferramenta de Busca Semântica para Técnicos")

question = st.text_input("Como posso ajudar?")

if question:
    results = search_best_practices(question, documents)
    st.write("Resultados encontrados:")
    for i, (filename, doc_content, filepath, pages, similarity) in enumerate(results):
        st.write(f"Resultado {i + 1}: Similaridade: {similarity:.4f}")
        st.write(f"Documento: {filename}")
        
        snippets = extract_relevant_snippets(question, doc_content, pages)
        for snippet, page in snippets:
            st.write(f"- {snippet} (Página {page})")
            
            if filepath.endswith(".pdf"):
                page_image_buffer = get_pdf_page_image(filepath, page)  # Exibir a página relevante
                if page_image_buffer:
                    st.image(page_image_buffer, caption=f"Imagem da página {page} do documento {filename}")
