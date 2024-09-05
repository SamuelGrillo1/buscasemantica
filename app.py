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


@st.cache_resource
def get_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

tokenizer, model = get_model_and_tokenizer()


def extract_text_from_file(file):
    text = ""
    pages = []
    if file.type == "text/plain":
        text = file.read().decode('utf-8')
    elif file.type == "application/pdf":
        text, pages = extract_text_from_pdf(file)
    return text, pages


def extract_text_from_pdf(file):
    text = ""
    pages = []
    try:
        with pdfplumber.open(file) as pdf:
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
        results.append((doc['filename'], doc['content'], doc['pages'], cosine_similarities[idx]))
    return results


def extract_relevant_snippets(query, document, pages, top_n=3):
    sentences = nltk.sent_tokenize(document)
    query_embedding = get_bert_embeddings(query)
    sentence_embeddings = torch.vstack([get_bert_embeddings(sentence) for sentence in sentences])
    
    similarities = calculate_similarity(query_embedding, sentence_embeddings)
    top_indices = similarities.argsort()[::-1][:top_n]
    
    snippets = [(sentences[i], pages[i]) for i in top_indices] if pages else [(sentences[i], None) for i in top_indices]
    return snippets


def get_pdf_page_image(file, page_number):
    try:
        with pdfplumber.open(file) as pdf:
            page = pdf.pages[page_number - 1]
            image = page.to_image(resolution=200)

            img_buffer = BytesIO()
            image.original.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            return img_buffer
    except Exception as e:
        st.error(f"Erro ao obter a imagem da página: {e}")
        return None



st.title("Ferramenta de Busca Semântica para Técnicos")

uploaded_files = st.file_uploader("Upload de documentos", accept_multiple_files=True, type=["txt", "pdf"])

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        text, pages = extract_text_from_file(uploaded_file)
        if text:
            tokens = nltk.word_tokenize(text)
            documents.append({
                'filename': uploaded_file.name,
                'content': " ".join(tokens),
                'pages': pages
            })

    question = st.text_input("Como posso ajudar?")
    
    if question and documents:
        results = search_best_practices(question, documents)
        st.write("Resultados encontrados:")
        for i, (filename, doc_content, pages, similarity) in enumerate(results):
            st.write(f"Resultado {i + 1}: Similaridade: {similarity:.4f}")
            st.write(f"Documento: {filename}")
            
            snippets = extract_relevant_snippets(question, doc_content, pages)
            for snippet, page in snippets:
                st.write(f"- {snippet} (Página {page})")
                
                if page and filename.endswith(".pdf"):
                    page_image_buffer = get_pdf_page_image(uploaded_file, page)
                    if page_image_buffer:
                        st.image(page_image_buffer, caption=f"Imagem da página {page} do documento {filename}")
