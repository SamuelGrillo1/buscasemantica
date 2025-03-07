import os
import nltk
import streamlit as st
import numpy as np
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from io import BytesIO
from PIL import Image


nltk.download('punkt')


@st.cache_resource
def get_model_and_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

tokenizer, model = get_model_and_tokenizer()


def load_documents_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filepath.endswith(".txt") or filepath.endswith(".pdf"):
            text, pages = extract_text_from_file(filepath)
            if text:
                documents.append({
                    'filename': filename,
                    'content': text,  
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
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


def search_best_practices(query, documents):
    query_embedding = get_bert_embeddings(query)
    results = []
    

    for doc in documents:
        doc_embedding = get_bert_embeddings(doc['content'])
        similarity = calculate_similarity(query_embedding, doc_embedding)
        results.append((doc['filename'], doc['content'], doc['pages'], similarity))
    

    results.sort(key=lambda x: x[3], reverse=True)
    return results[:5]  


def calculate_similarity(query_embedding, doc_embedding):
    return cosine_similarity(query_embedding.detach().numpy(), doc_embedding.detach().numpy()).flatten()[0]


def extract_relevant_snippets(query, document, pages, top_n=3):
    sentences = nltk.sent_tokenize(document)
    query_embedding = get_bert_embeddings(query)
    
 
    sentence_embeddings = [get_bert_embeddings(sentence) for sentence in sentences]
    similarities = [calculate_similarity(query_embedding, embedding) for embedding in sentence_embeddings]
    
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    snippets = [(sentences[i], pages[i] if pages else None) for i in top_indices]
    return snippets


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


st.title("Ferramenta de Busca Semântica para Técnicos")


docs_directory = "Dataset_BS"
documents = load_documents_from_directory(docs_directory)

if documents:
    question = st.text_input("Como posso ajudar?")
    
    if question:

        results = search_best_practices(question, documents)
        st.write("Resultados encontrados:")
        for i, (filename, doc_content, pages, similarity) in enumerate(results):
            st.write(f"Resultado {i + 1}: Similaridade: {similarity:.4f}")
            st.write(f"Documento: {filename}")
            
            snippets = extract_relevant_snippets(question, doc_content, pages)
            for snippet, page in snippets:
                st.write(f"- {snippet} (Página {page})")
                
                if page and filename.endswith(".pdf"):
                    page_image_buffer = get_pdf_page_image(os.path.join(docs_directory, filename), page)
                    if page_image_buffer:
                        st.image(page_image_buffer, caption=f"Imagem da página {page} do documento {filename}")
