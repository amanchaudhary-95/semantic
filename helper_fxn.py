import streamlit as st
import time
import faiss
from sentence_transformers import SentenceTransformer
import json

@st.cache_data(show_spinner='Loading Data.....')
def read_json():
    with open('db.json','r') as f:
        final_data = json.load(f)
    return final_data

@st.cache_data(show_spinner='Loading Word Embedding...')
def read_faiss(choose_model):
    index = faiss.read_index(f'./index/{st.session_state["models"][choose_model]}_raw_sementic.index')
    return index

@st.cache_resource(show_spinner='Loading the model...')
def load_model(choose_model):
    model = SentenceTransformer(f'./sementic/{st.session_state["models"][choose_model]}-finetuned-raw').to('cuda')
    return model

@st.dialog('🚀 Welcome', width='large')
def popup():
    st.success(f"Hi **Guest** ! Welcome to the Sementic Search Engine App", icon=':material/waving_hand:')
    st.info(f"""
               **Student Name:** Aman Chaudhary  
                **Enrollment:** 2022aa05016  
               **Degree:** M.Tech in Artificial Intelligence and Machine Learning  
               **Dissertation Topic:** Development of a semantic search engine to find relevant documents from a corpus of scanned documents   
               **Supervisor:** Sumit Chaudhary  
               **Mentor:** Prof. Vijayalakshmi Anand""", icon=':material/info:')

def search(index, query_vector, top_k=10, choose_model = None):
    t_in = time.time()
    top_k_arr= index.search(query_vector, top_k)
    final_data = read_json()
    a, top_k_ids = top_k_arr[0][0], top_k_arr[1][0]
    if choose_model == 'T5':
        results = [final_data[top_k_ids[i]-1] for i in range(top_k) if 100*abs(a[0] - a[i]) < 10]
    else:
        results = [final_data[top_k_ids[i]-1] for i in range(top_k) if abs(a[0] - a[i]) < 10]
    t_out = time.time()
    return results, t_out-t_in
