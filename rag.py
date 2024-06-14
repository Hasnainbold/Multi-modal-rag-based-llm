from langchain_openai import ChatOpenAI
import requests
import openai
from streamlit_feedback import streamlit_feedback
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from langchain.prompts import PromptTemplate
from weaviate.embedded import EmbeddedOptions
import os
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatHuggingFace
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
from sentence_transformers import CrossEncoder
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    ContextRelevancy,
    answer_correctness,
    answer_similarity
)
from uuid import uuid4
import streamlit as st
from sentence_transformers import SentenceTransformer
from ragas import evaluate
from typing import Sequence
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableLambda
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.text_splitter import *
from langchain.smith import RunEvalConfig
from langchain_core.runnables import chain
from langsmith import Client
import re
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from openai import OpenAI
from transformers import pipeline
import torch
from Vector_database import VectorDatabase
from Rag_chain import RAGEval
from Query_agent import *
from langsmith.run_trees import RunTree


# Embedding Model
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True).tolist()


# Parser
class MistralParser:
    stopword = 'Answer:'
    parser = ''

    def __init__(self):
        self.parser = StrOutputParser()

    def invoke(self, query):
        ans = self.parser.invoke(query)
        return ans[ans.find(self.stopword)+len(self.stopword):].strip()


# Caching models
@st.cache_resource(show_spinner=False)
def load_bi_encoder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2", model_kwargs={"device": "cpu"})


@st.cache_resource(show_spinner=False)
def pine_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")  # 784 dimension + euclidean


@st.cache_resource(show_spinner=False)
def weaviate_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_cross():
    return CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, device="cpu")


@st.cache_resource(show_spinner=False)
def pine_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, device="cpu")


@st.cache_resource(show_spinner=False)
def weaviate_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device="cpu")


@st.cache_resource(show_spinner=False)
def load_chat_model():
    template = '''
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question accurately.
    Question: {question}
    Context: {context}
    Answer:
    '''
    return HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512, "query_wrapper_prompt": template}
    )


@st.cache_resource(show_spinner=False)
def load_q_model():
    return HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
    )


bi_encoder = load_bi_encoder()
chat_model = load_chat_model()
cross_model = load_cross()

file_path = "software_data.txt"
url = st.secrets["WEAVIATE_URL"]
v_key = st.secrets["WEAVIATE_V_KEY"]
gpt_key = st.secrets["GPT_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

# Vector Database objects
pine_embed = pine_embedding_model()
weaviate_embed = weaviate_embedding_model()
pine_cross = pine_cross_encoder()
weaviate_cross = weaviate_cross_encoder()
pine_vb = VectorDatabase(pine_embed, pine_cross, 'Pinecone', st.secrets["PINECONE_API_KEY"], index='rag3', dimension=768, metric='euclidean', url=None)
weaviate_vb = VectorDatabase(weaviate_embed, weaviate_cross, 'Weaviate', st.secrets["WEAVIATE_V_KEY"], index=None, dimension=None, metric=None, url=url)
pine_text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)
weaviate_text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)
vb_list = [
    (pine_vb, pine_text_splitter),
    (weaviate_vb, weaviate_text_splitter)
]

mistral_parser = RunnableLambda(MistralParser().invoke)

q_model = load_q_model()
# q_parser = RunnableLambda(lambda ans: ans.split('\n')[-1].strip()[len('Output: '):])
alt_parser = RunnableLambda(lambda x: x[x.find('1. '):])
query_agent = RunnableLambda(QueryAgent(vb_list, q_model, cross_model, alt_parser).query)


# file_path, vb_list, cross_model
req = RAGEval(file_path, vb_list, cross_model)  # file_path,url,vb_key,gpt_key):
req.model_prep(chat_model, mistral_parser)  # model details
req.query_agent_prep(q_model, alt_parser)

if "run_id" not in st.session_state:
    st.session_state.run_id = uuid4()
if 'messages' not in st.session_state:
    st.session_state.messages = []

client = Client(api_url=st.secrets["LANGSMITH_URL"], api_key=st.secrets["LANGSMITH_API_KEY"])
mes = [message["role"]+": "+message["content"] for message in st.session_state.messages]
print(f"Run_ID -> {st.session_state.run_id}, {mes}")

# memory = ConversationBufferMemory(
#     chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
#     return_messages=True,
#     memory_key="chat_history"
# )

st.title('RAG Bot')
st.subheader('Converse with our Chatbot')
st.text("Some sample questions to ask:")
st.markdown("- What are adjustment points in the context of using a microscope, and why are they important?")
st.markdown("- What does alignment accuracy refer to, and how is it achieved in a microscopy context?")
st.markdown("- What are alignment marks, and how are they used in the alignment process?")
st.markdown("- What is the alignment process in lithography, and how does eLitho facilitate this procedure?")
st.markdown("- What can you do with the insertable layer in Smart FIB?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
rt = RunTree(
    name="RAG RunTree",
    run_type="chain",
    inputs={"messages": [message["role"]+": "+message["content"] for message in st.session_state.messages]},
    id=st.session_state.run_id
)


def fbcb():
    message_id = len(st.session_state.messages) - 1
    if message_id >= 0:
        st.session_state.messages[message_id]["feedback"] = st.session_state.fb_k
    f = open('feedback.txt', 'r+')
    f.write("FEEDBACK FOR : \n")
    f.write(prompt+'\n')
    print(st.session_state.fb_k)
    print(st.session_state.fb_k["score"])
    fb = "NEGATIVE -> " if st.session_state.fb_k["score"] == '👎' else "POSITIVE -> "
    for message in st.session_state.messages:
        f.write(message["content"]+'\n')
        f.write(fb)
        if st.session_state.fb_k['text'] is None:
            st.session_state.fb_k['text'] = ""
        f.write(st.session_state.fb_k['text']+'\n')
    f.close()
    with open('feedback.txt', 'r') as f:
        feed = f.read()
    client.create_feedback(
        run_id=st.session_state.run_id,
        key="fb_k",
        score=1 if fb == "Positive" else -1,
        feedback_source_type="model",
        comment=feed
    )


if prompt := st.chat_input("What's up?"):
    # feedback_option = "thumbs" if st.toggle(label="`Thumbs` ⇄ `Faces`", value=False) else "faces"
    st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = req.query(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.form('form'):
        streamlit_feedback(feedback_type="thumbs", align="flex-start", key='fb_k', optional_text_label="[Optional] Please provide an explanation")
        submit_button = st.form_submit_button('Save feedback', on_click=fbcb)
        if not submit_button:
            print('Click the Submit button')

with open('feedback.txt', 'r') as fd:
    rt.end(outputs={'outputs': fd.read()})
rt.post()


def reset_conversation():
    st.session_state.messages = []


st.button('Reset Chat', on_click=reset_conversation)
