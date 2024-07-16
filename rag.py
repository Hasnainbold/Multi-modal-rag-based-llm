from streamlit_feedback import streamlit_feedback
import streamlit as st
from uuid import uuid4
import unstructured
import pdfminer
import torch.nn.functional as F
import matplotlib.pyplot as plt
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.schema import ImageNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
import requests
from PIL import Image
import fitz
from unidecode import unidecode
import weaviate.classes as wvc
import multiprocessing
from langchain.document_loaders import TextLoader, JSONLoader
from langchain.docstore.document import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate, Pinecone
import weaviate
import asyncio
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from weaviate.embedded import EmbeddedOptions
import os
from os import listdir
from zipfile import ZipFile
from os.path import isfile, join
import torchvision.transforms as T
from pathlib import Path
from pprint import pprint
from sklearn.cluster import DBSCAN
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatHuggingFace
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
from scipy.spatial.distance import euclidean
from sentence_transformers import CrossEncoder, SentenceTransformer
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    ContextRelevancy,
    answer_correctness,
    answer_similarity
)
from ragas import evaluate
from typing import Sequence, List
import pandas as pd
import numpy as np
import json
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableLambda, chain
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.text_splitter import *
from langchain.smith import RunEvalConfig
from langsmith import Client
import re
import shutil
import io
import pytesseract
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph, Graph, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from openai import OpenAI
from transformers import (pipeline, AutoFeatureExtractor, AutoModel, AutoProcessor,
                          AutoModelForVision2Seq, AutoImageProcessor)
import torch
from langchain.retrievers import ContextualCompressionRetriever, MergerRetriever
from langchain_community.document_compressors import LLMLinguaCompressor
from typing_extensions import TypedDict
import base64
from io import BytesIO
import pyarrow as pa
import pyarrow.dataset as ds
from Rag_chain import *
from Query_agent import *
from langsmith.run_trees import RunTree
from Databases import *

showWarningOnDirectExecution = False


@st.cache_resource(show_spinner=False)
def settings():
    return HuggingFaceEmbedding(model_name="BAAI/bge-base-en")


# Embedding Model
class SentenceTransformerEmbeddings:
  """
    Wrapper Class for SentenceTransformer Class
  """
  def __init__(self, model_name: str):
    """
      Initiliases a Sentence Transformer
    """
    self.model = SentenceTransformer(model_name)

  def embed_documents(self, texts):
    """
    Returns a list of embeddings for the given texts.
    """
    return self.model.encode(texts, convert_to_tensor=True).tolist()

  def embed_query(self, text):
    """
      Returns a list of embeddings for the given text.
    """
    return self.model.encode(text, convert_to_tensor=True).tolist()


# Parser
class MistralParser:
  """
    Wrapper Class for StrOutputParser Class
    Custom made for Mistral models
  """
  stopword = 'Answer:'

  def __init__(self):
    """
      Initiliases a StrOutputParser as the base parser
    """
    self.parser = StrOutputParser()

  def invoke(self,query):
    """
      Invokes the parser and finds the Model response
    """
    ans = self.parser.invoke(query)
    return ans[ans.find(self.stopword)+len(self.stopword):].strip()


class ChatGPT:
  """
    Wrapper Class for ChatGPT Class
  """

  def __init__(self,model,api_key, template):
    self.model = model
    self.api_key = api_key
    self.client = OpenAI(api_key=api_key)
    self.template = template

  def image(self,image):
    """
      Image to Text Conversion
    """

    def pil_to_base64(img):
      buffered = BytesIO()
      img.save(buffered, format="PNG")
      img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
      return img_str

    image_str = pil_to_base64(image)
    headers = {"Content-Type": "application/json","Authorization": f"Bearer {self.api_key}"}
    payload = {"model": self.model,
    "messages": [{"role": "user",
       "content": [{"type": "text","text": self.template},{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image_str}"}}]
      }
    ],"max_tokens": 20}
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res = response.json()
    if 'error' not in res:
      return res['choices'][0]['message']['content']
    else:
      return res['error']['message']

  def chat(self, prompt):
    """
      Text Conversation
    """
    message = [{"role":"user", "content":prompt.messages[0].content}]
    return self.client.chat.completions.create(messages=message, model=self.model).choices[0].message.content


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


@st.cache_resource(show_spinner=False)
def vector_database_prep():
    # Vector Database objects
    i_model = vision_model
    pinecone_embed = pine_embedding_model()
    weaviate_embed = weaviate_embedding_model()

    vb1 = UnifiedDatabase('vb1', 'lancedb/rag')
    vb1.model_prep(extractor, i_model, weaviate_embed,
                   RecursiveCharacterTextSplitter(chunk_size=1330, chunk_overlap=35))
    vb2 = UnifiedDatabase('vb2', 'lancedb/rag')
    vb2.model_prep(extractor, i_model, pinecone_embed,
                   RecursiveCharacterTextSplitter(chunk_size=1330, chunk_overlap=35))
    vb_list = [vb1, vb2]

    with open('software_final.txt', 'r') as f:  # texts
        data = f.read()

    image_path = './figures'
    image_info = []
    for image_file in os.listdir(image_path):
        if image_file.endswith('.jpg'):
            image_info.append({
                "image_file_name": image_file,
                "image": Image.open(os.path.join(image_path, image_file)),
                "pg_no": int(image_file.split('-')[1])
            })

    image_content = []
    pdf_path = './Software Manual SmartFIB v1.2.pdf'
    with fitz.open(pdf_path) as pdf_file:
        for image in image_info:
            text = pdf_file[image['pg_no']].get_text()
            image_content.append((
                image['image_file_name'],
                text + '\n' + image['image_file_name'],
                image['image']
            ))

    for vb in vb_list:
        vb.upsert(data)
        vb.upsert(image_content)  # image_cont = dict[image_file_path, context, PIL]
    return vb_list


@st.cache_resource(show_spinner=False)
def image_dataset_creation(file_path):
  def image_to_bytes(image):
    if image.mode == 'CMYK':
        image = image.convert('RGB')
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()

  content = []
  headers = [{
      'page_index': 0,
      'block_index': 0,
      'line_index': 0,
      'span_index': 0,
      'text': 'Software Manual SmartFIB Application Software for Crossbeam Workstations',
      'content': []}]

  page_cont = dict()
  with fitz.open('Software Manual SmartFIB v1.2.pdf') as pdf_file:
    for page_index in range(len(pdf_file)):
      page_cont[page_index] = pdf_file[page_index].get_text('dict')

  for page_index in page_cont:
    page_dict = page_cont[page_index]
    blocks = page_dict['blocks']
    text_blocks = [block for block in blocks if block['type'] == 0]
    for text_block in text_blocks:
      lines = text_block['lines']
      for line in range(len(lines)):
        spans = lines[line]['spans']
        for span in spans:
          if span['size'] >= 10.4 and 'bol' in span['font'].lower():
            new_header = {
                'page_index': page_index,
                'block_index': text_block['number'],
                'line_index': line,
                'span_index': spans.index(span),
                'text': span['text'],
                'content': []
            }
            headers[-1]['content'] = content
            headers.append(new_header)
            content = []
          else:
            content.append(span['text'])
  # header_df = pd.DataFrame(headers)

  with fitz.open(file_path) as pdf_file:
    per_page_text = [pdf_file[page_index].get_text() for page_index in range(len(pdf_file))]
    # iterate over pdf pages
    images = []
    for page_index in range(len(pdf_file)):  # which page in the doc
        # get the page itself
        page = pdf_file[page_index]
        image_list = page.get_images()
        # printing number of images found in this page
        if image_list:
            for image_index, img in enumerate(image_list, start=1):  # which image in the current page
              xref = img[0]  # get the XREF of the image
              base_image = pdf_file.extract_image(xref)
              image_bytes = base_image["image"]  # extract the image bytes
              image_ext = base_image["ext"]  # get the image extension
              image = Image.open(io.BytesIO(image_bytes))  # load it to PIL
              image_content = per_page_text[0] + "\n" + pytesseract.image_to_string(image)
              header_names = [i['text'].replace('/','\\') for i in headers if i['page_index'] == page_index and not i['text'].isnumeric()]
              if len(header_names) == 0:
                header_names = [pytesseract.image_to_string(image) + ' ' + image_content[:image_content.find(' ')] + '_' + str(page_index)]
              image_dict = {
                  "page_index": page_index,
                  "image_index": image_index,
                  "image_ext": image_ext,
                  "image": image,
                  "image_file_name": ",".join(header_names) + '_' + str(image_index),
                  "image_content": image_content
              }
              images.append(image_dict)
  im_df = pd.DataFrame(images)
  im_df['image'] = im_df['image'].apply(image_to_bytes)
  dataset = Dataset(pa.Table.from_pandas(im_df))
  return dataset


@st.cache_resource(show_spinner=False)
def load_image_model(model):
    extractor = AutoFeatureExtractor.from_pretrained(model)
    im_model = AutoModel.from_pretrained(model)
    return extractor, im_model

@st.cache_resource(show_spinner=False)
def load_nomic_model():
    return  AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5"), AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5",
                                         trust_remote_code=True)


Settings.embed_model = settings()
processor, vision_model = load_nomic_model()
bi_encoder = load_bi_encoder()
chat_model = load_chat_model()
cross_model = load_cross()
extractor, image_model = load_image_model("google/vit-base-patch16-224-in21k")

file_path = "software_data.txt"
url = st.secrets["WEAVIATE_URL"]
v_key = st.secrets["WEAVIATE_V_KEY"]
gpt_key = st.secrets["GPT_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["OPENAI_API_KEY"] = gpt_key
f_url = st.secrets['FEEDBACK_URL']
f_api = st.secrets['FEEDBACK_API']
feedback_file = "feedback_loop.txt"
im_db_url = st.secrets["IMAGE_URL"]
im_db_key = st.secrets["IMAGE_API"]
image_file = 'Software Manual SmartFIB v1.2.pdf'
image_folder = './figures'
txt_db_url = st.secrets["TEXT_URL"]
txt_db_key = st.secrets["TEXT_API"]

fd = False
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
mistral_parser = RunnableLambda(MistralParser().invoke)
vb_list = vector_database_prep()
q_model = load_q_model()
alt_parser = RunnableLambda(lambda x: x[x.find('1. '):])
gpt_model = RunnableLambda(ChatGPT("gpt-4o", api_key=gpt_key, template="""You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question accurately.
    Question: {question}
    Context: {context}
    Answer:""").chat)
gq_model = RunnableLambda(ChatGPT('gpt-3.5-turbo', api_key=gpt_key, template="""You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question accurately.
    Question: {question}
    Context: {context}
    Answer:""").chat)
dataset = image_dataset_creation(image_file)
weaviate_embed = weaviate_embedding_model()
pine_embed = pine_embedding_model()
feedback_db = TextDatabase('feedback', 'lancedb/rag')
feedback_db.model_prep(weaviate_embed, RecursiveCharacterTextSplitter(chunk_size=1330, chunk_overlap=35))
with open('feedback_loop.txt', 'r') as f:
  feedback = f.read()
feedback_db.upsert(feedback)

req = RAGEval(vb_list, cross_model)
req.model_prep(chat_model, mistral_parser)
req.query_agent_prep(q_model, alt_parser)
req.feedback_prep(uri='lancedb/rag', table_name='feedback',
                  file=feedback_file, embedder=weaviate_embed,
                  splitter=RecursiveCharacterTextSplitter(chunk_size=1330, chunk_overlap=35))

if "run_id" not in st.session_state:
    st.session_state.run_id = uuid4()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'image' not in st.session_state:
    st.session_state['image'] = []
if 'conv_id' not in st.session_state:
    st.session_state['conv_id'] = {}

client = Client(api_url=st.secrets["LANGSMITH_URL"], api_key=st.secrets["LANGSMITH_API_KEY"])
mes = []
for message in st.session_state.messages:
    if type(message['content']) is dict:
        mes.append(message['role']+": "+message["content"]["text"])
    else:
        mes.append(message['role'] + ": " + message["content"])
print(f"Run_ID -> {st.session_state.run_id}, {mes}")

st.title('RAG Bot')
st.subheader('Converse with our Chatbot')
st.markdown("You may input text or image")
st.markdown("Some sample questions to ask:")
st.markdown("- What are adjustment points in the context of using a microscope, and why are they important?")
st.markdown("- What does alignment accuracy refer to, and how is it achieved in a microscopy context?")
st.markdown("- What are alignment marks, and how are they used in the alignment process?")
st.markdown("- What is the alignment process in lithography, and how does eLitho facilitate this procedure?")
st.markdown("- What can you do with the insertable layer in Smart FIB?")

open('feedback.txt', 'w').close()

rt = RunTree(
    name="RAG RunTree",
    run_type="chain",
    inputs={"messages": mes},
    id=st.session_state.run_id
)


def fbcb():
    if st.session_state.fb_k is None:
        st.session_state.fb_k = {'type': 'thumbs', 'score': '👎', 'text': ''}
    message_id = len(st.session_state.messages) - 1
    if message_id >= 0:
        st.session_state.messages[message_id]["feedback"] = st.session_state.fb_k

    s = f"The feedback for {prompt} "
    fb = "NEGATIVE " if st.session_state.fb_k["score"] == '👎' else "POSITIVE "
    for _ in st.session_state.messages:
        if st.session_state.fb_k['text'] is None:
            st.session_state.fb_k['text'] = ""
        s += f'is {fb} and the response is '
        if fb == "NEGATIVE ":
            s += st.session_state.fb_k['text']
        else:
            fsa = [d['content'] for d in st.session_state.messages if d["role"] == 'assistant']
            if isinstance(fsa[-1], str):
                s += fsa[-1]
            else:
                s += fsa[-1]['text']
        s += '\n'
    with open('feedback.txt', 'r+') as fd:  # feedback records all feedback for this run
        fd.write(s)
    with open('feedback_loop.txt', 'r+') as fd:  # feedback loop records feedback for all runs
        fd.write(s)
    feedback_db.upsert(s)
    with open('feedback.txt', 'r') as fd:
        feed = fd.read()
    client.create_feedback(
        run_id=st.session_state.run_id,
        key="fb_k",
        score=1 if fb == "Positive" else -1,
        feedback_source_type="model",
        comment=feed
    )


def plot_images(images_path, output_path, image_name, top_k=5):
    images_shown = 0

    plt.figure(figsize=(16, 9))
    for img_path in images_path:
        if os.path.isfile(img_path):
            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(Image.open(img_path))
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= top_k:
                break

    # Ensure the directory exists
    os.makedirs(output_path, exist_ok=True)
    full_image_path = os.path.join(output_path, image_name)
    plt.savefig(full_image_path)
    plt.close()  # Close the plot to free up memory
    return full_image_path


all_images = []
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    result = req.query(Image.open(uploaded_file), 5)  # dict
    images = result['image']  # list
    ai_response = "Context for the image: " + "".join(result['text'])  # str
    conv_id = uuid.uuid4()
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.session_state['image'] += images
    st.session_state['conv_id'][conv_id] = {
        "user_messages": {},
        "ai_messages": {"role": "assistant", "content": ai_response},
        "images": images  # list
    }

if prompt := st.chat_input("What's up?"):
    fd = True
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = req.query(prompt, 5)  # prompt is a str
    images = response['image']
    st.session_state.messages.append({"role": "assistant", "content": response['text']})

    conv_id = uuid.uuid4()
    st.session_state['conv_id'][conv_id] = {
        "user_messages": {"role": "user", "content": prompt},
        "ai_messages": {"role": "assistant", "content": response['text']},
        "images": images
    }

for conv_id in st.session_state['conv_id']:
    dic = st.session_state['conv_id'][conv_id]
    user_messages, ai_messages, images = dic['user_messages'], dic['ai_messages'], dic['images']
    if len(user_messages) > 0:
        with st.chat_message(user_messages["role"]):
            st.text(user_messages["content"])
    with st.chat_message(ai_messages["role"]):
        # if type(ai_messages['content']) is dict:
        #     st.text(ai_messages['content']['text'])
        #     if len(ai_messages['content']['image']) > 0:
        #         for image in ai_messages['content']['image']:
        #             if os.path.isfile(image):
        #                 st.text(image)
        #                 st.image(Image.open(Image.open(os.path.join('./figures', image)), use_column_width=True))
        #         # image_name = './rag_result_' + str(uuid.uuid4()) + '.jpg'
        #         # output_path = './output_images'
        #         # saved_image_path = plot_images(ai_messages['content']['image'], output_path, image_name)
        #         # st.image(Image.open(saved_image_path), use_column_width=True)
        # else:
        st.text(ai_messages["content"])
    if len(images) > 0:
        for image in images:
            st.text(image)
            st.image(Image.open(os.path.join('./figures', image)), use_column_width=True)

if fd:
    with st.form('form'):
        streamlit_feedback(feedback_type="thumbs", align="flex-start",
                           key='fb_k', optional_text_label="[Optional] Please provide an explanation")
        submit_button = st.form_submit_button('Save feedback', on_click=fbcb)
        if not submit_button:
            print('Click the Submit button')

with open('feedback.txt', 'r')as f:
    rt.end(outputs={'outputs': f.read()})
rt.post()


def reset_conversation():
    st.session_state.messages = []
    st.session_state['image'] = []
    st.session_state['conv_id'] = []


st.button('Reset Chat', on_click=reset_conversation)

