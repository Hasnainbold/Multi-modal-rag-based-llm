import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory, ConversationBufferMemory
from streamlit_feedback import streamlit_feedback
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
# from langchain_openai import ChatOpenAI
import requests
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
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
)
from sentence_transformers import SentenceTransformer
from ragas.metrics import faithfulness, answer_correctness, answer_similarity
from ragas import evaluate
from typing import Sequence
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableLambda
# from pinecone_notebooks.colab import Authenticate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.text_splitter import *
from langchain.smith import RunEvalConfig
from langchain_core.runnables import chain
from langsmith import Client
# from operator import itemgetter
# import textstat
from langsmith.schemas import Run, Example
from Vector_database import VectorDatabase
from Rag_chain import RAGEval


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

  def invoke(self,query):
    ans = self.parser.invoke(query)
    return ans[ans.find(self.stopword)+len(self.stopword):].strip()

# Caching models
@st.cache_resource
def load_bi_encoder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2",model_kwargs={"device": "cpu"})

@st.cache_resource
def pine_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2") # 784 dimension + euclidean

@st.cache_resource
def weaviate_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_cross():
    return CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, device="cpu")

@st.cache_resource
def pine_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, device="cpu")

@st.cache_resource
def weaviate_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device="cpu")

@st.cache_resource
def load_chat_model():
    template = '''
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question accurately.
    If you don't know the answer, just say that you don't know.
    Question: {question}
    Context: {context}
    Answer:
    '''
    return HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":512, "query_wrapper_prompt":template}
)


bi_encoder = load_bi_encoder()
chat_model = load_chat_model()
cross_model = load_cross()

file_path = "software_data.txt"
url = st.secrets["WEAVIATE_URL"]
v_key = st.secrets["WEAVIATE_V_KEY"]
gpt_key = st.secrets["GPT_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Vector Database objects
pine_embed = pine_embedding_model()
weaviate_embed = weaviate_embedding_model()
pine_cross = pine_cross_encoder()
weaviate_cross = weaviate_cross_encoder()
pine_vb = VectorDatabase(pine_embed, pine_cross, 'Pinecone', st.secrets["PINECONE_API_KEY"], index='rag', dimension=768, metric='euclidean', url=None)
weaviate_vb = VectorDatabase(weaviate_embed, weaviate_cross, 'Weaviate', st.secrets["WEAVIATE_V_KEY"], index=None, dimension=None, metric=None, url=url)
pine_text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)
weaviate_text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)
vb_list = [
    (pine_vb, pine_text_splitter),
    (weaviate_vb, weaviate_text_splitter)
]

mistral_parser = RunnableLambda(MistralParser().invoke)

# file_path, vb_list, cross_model
re = RAGEval(file_path, vb_list, cross_model)  # file_path,url,vb_key,gpt_key):
re.model_prep(chat_model, mistral_parser)  # model details

client = Client(api_url=st.secrets["LANGSMITH_URL"], api_key=st.secrets["LANGSMITH_API_KEY"])

memory = ConversationBufferMemory(
    chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
    return_messages=True,
    memory_key="chat_history",
)

st.title('RAG Bot')
st.subheader('Converse with our Chatbot')
st.text("Some sample questions to ask:")
st.markdown("- What are adjustment points in the context of using a microscope, and why are they important?")
st.markdown("- What does alignment accuracy refer to, and how is it achieved in a microscopy context?")
st.markdown("- What are alignment marks, and how are they used in the alignment process?")
st.markdown("- What is the alignment process in lithography, and how does eLitho facilitate this procedure?")
st.markdown("- What can you do with the insertable layer in Smart FIB?")

if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    feedback_option = "faces" if st.toggle(label="`Thumbs` ⇄ `Faces`", value=False) else "thumbs"

    st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = re.query(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    if st.session_state.get("run_id"):
        feedback = streamlit_feedback(
            feedback_type=feedback_option,  # Apply the selected feedback style
            optional_text_label="[Optional] Please provide an explanation",  # Allow for additional comments
            key=f"feedback_{st.session_state.run_id}",
        )

def reset_conversation():
  st.session_state.messages = []


st.button('Reset Chat', on_click=reset_conversation)
