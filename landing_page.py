import shutil
import streamlit as st
st.set_page_config(
   page_title="RAG Configuration",
   page_icon="🤖",
   layout="wide",
   initial_sidebar_state="collapsed"
)
import re
import os
import spire.pdf
import fitz
from src.Databases import *
from langchain.text_splitter import *
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import (AutoFeatureExtractor, AutoModel, AutoImageProcessor)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


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


@st.cache_resource(show_spinner=False)
def settings():
    return HuggingFaceEmbedding(model_name="BAAI/bge-base-en")


@st.cache_resource(show_spinner=False)
def pine_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")  # 784 dimension + euclidean


@st.cache_resource(show_spinner=False)
def weaviate_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_image_model(model):
    extractor = AutoFeatureExtractor.from_pretrained(model)
    im_model = AutoModel.from_pretrained(model)
    return extractor, im_model


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
    If the question is not related to the context, just answer 'I don't know'.
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
def load_image_model(model):
    extractor = AutoFeatureExtractor.from_pretrained(model)
    im_model = AutoModel.from_pretrained(model)
    return extractor, im_model


@st.cache_resource(show_spinner=False)
def load_nomic_model():
    return  AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5"), AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5",
                                         trust_remote_code=True)


@st.cache_resource(show_spinner=False)
def vector_database_prep(file):
    def data_prep(file):
        def findWholeWord(w):
            return re.compile(r'\b{0}\b'.format(re.escape(w)), flags=re.IGNORECASE).search

        file_name = file.name
        pdf_file_path = os.path.join(os.getcwd(), 'pdfs', file_name)
        image_folder = os.path.join(os.getcwd(), f'figures_{file_name}')
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # everything down here is wrt pages dir
        print('1. folder made')
        with spire.pdf.PdfDocument() as doc:
            doc.LoadFromFile(pdf_file_path)
            images = []
            for page_num in range(doc.Pages.Count):
                page = doc.Pages[page_num]
                for image_num in range(len(page.ImagesInfo)):
                    imageFileName = os.path.join(image_folder, f'figure-{page_num}-{image_num}.png')
                    image = page.ImagesInfo[image_num]
                    image.Image.Save(imageFileName)
                    images.append({
                        "image_file_name": imageFileName,
                        "image": image
                    })
        print('2. image extraction done')
        image_info = []
        for image_file in os.listdir(image_folder):
            if image_file.endswith('.png'):
                image_info.append({
                    "image_file_name": image_file[:-4],
                    "image": Image.open(os.path.join(image_folder, image_file)),
                    "pg_no": int(image_file.split('-')[1])
                })
        print('3. temporary')
        figures = []
        with fitz.open(pdf_file_path) as pdf_file:
            data = ""
            for page in pdf_file:
                text = page.get_text()
                if not (findWholeWord('table of contents')(text) or findWholeWord('index')(text)):
                    data += text
            data = data.replace('}', '-')
            data = data.replace('{', '-')
            print('4. Data extraction done')
            hs = []
            for i in image_info:
                src = i['image_file_name'] + '.png'
                headers = {'_': []}
                header = '_'
                page = pdf_file[i['pg_no']]
                texts = page.get_text('dict')
                for block in texts['blocks']:
                    if block['type'] == 0:
                        for line in block['lines']:
                            for span in line['spans']:
                                if 'bol' in span['font'].lower() and not span['text'].isnumeric():
                                    header = span['text']
                                    print("header: ", header)
                                    headers[header] = [header]
                                else:
                                    headers[header].append(span['text'])
                                try:
                                    if findWholeWord('fig')(span['text']):
                                        i['image_file_name'] = span['text']
                                        figures.append(span['text'].split('fig')[-1])
                                    elif findWholeWord('figure')(span['text']):
                                        i['image_file_name'] = span['text']
                                        figures.append(span['text'].lower().split('figure')[-1])
                                    else:
                                        pass
                                except re.error:
                                    pass
                if not i['image_file_name'].endswith('.png'):
                    s = i['image_file_name'] + '.png'
                    i['image_file_name'] = s
                    os.rename(os.path.join(image_folder, src), os.path.join(image_folder, i['image_file_name']))
                hs.append({"image": i, "header": headers})
            print('5. header and figures done')
            figure_contexts = {}
            for fig in figures:
                figure_contexts[fig] = []
                for page_num in range(len(pdf_file)):
                    page = pdf_file[page_num]
                    texts = page.get_text('dict')
                    for block in texts['blocks']:
                        if block['type'] == 0:
                            for line in block['lines']:
                                for span in line['spans']:
                                    if findWholeWord(fig)(span['text']):
                                        print('figure mention: ', span['text'])
                                        figure_contexts[fig].append(span['text'])
            print('6. Figure context collected')
            contexts = []
            for h in hs:
                context = ""
                for q in h['header'].values():
                    context += "".join(q)
                s = pytesseract.image_to_string(h['image']['image'])
                qwea = context + '\n' + s if len(s) != 0 else context
                contexts.append((
                    h['image']['image_file_name'],
                    qwea,
                    h['image']['image']
                ))
            print('7. Overall context collected')
            image_content = []
            for fig in figure_contexts:
                for c in contexts:
                    if findWholeWord(fig)(c[0]):
                        s = c[1] + '\n' + "\n".join(figure_contexts[fig])
                        s = str("\n".join(
                            [
                                "".join([h for h in i.strip() if h.isprintable()])
                                for i in s.split('\n')
                                if len(i.strip()) != 0
                            ]
                        ))
                        image_content.append((
                            c[0],
                            s,
                            c[2]
                        ))
            print('8. Figure context added')

        return data, image_content

    # Vector Database objects
    extractor, i_model = st.session_state['extractor'], st.session_state['image_model']
    pinecone_embed = st.session_state['pinecone_embed']
    weaviate_embed = st.session_state['weaviate_embed']

    vb1 = UnifiedDatabase('vb1', 'lancedb/rag')
    vb1.model_prep(extractor, i_model, weaviate_embed,
                   RecursiveCharacterTextSplitter(chunk_size=1330, chunk_overlap=35))
    vb2 = UnifiedDatabase('vb2', 'lancedb/rag')
    vb2.model_prep(extractor, i_model, pinecone_embed,
                   RecursiveCharacterTextSplitter(chunk_size=1330, chunk_overlap=35))
    vb_list = [vb1, vb2]

    data, image_content = data_prep(file)
    for vb in vb_list:
        vb.upsert(data)
        vb.upsert(image_content)  # image_cont = dict[image_file_path, context, PIL]
    return vb_list


os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["OPENAI_API_KEY"] = st.secrets["GPT_KEY"]
st.session_state['pdf_file'] = []
st.session_state['vb_list'] = []
st.session_state['Settings.embed_model'] = settings()
st.session_state['processor'], st.session_state['vision_model'] = load_nomic_model()
st.session_state['bi_encoder'] = load_bi_encoder()
st.session_state['chat_model'] = load_chat_model()
st.session_state['cross_model'] = load_cross()
st.session_state['q_model'] = load_q_model()
st.session_state['extractor'], st.session_state['image_model'] = load_image_model("google/vit-base-patch16-224-in21k")
st.session_state['pinecone_embed'] = pine_embedding_model()
st.session_state['weaviate_embed'] = weaviate_embedding_model()

st.title('Multi-modal RAG based LLM for Information Retrieval')
st.subheader('Converse with our Chatbot')
st.markdown('Enter a pdf file as a source.')
uploaded_file = st.file_uploader("Choose an pdf document...", type=["pdf"], accept_multiple_files=False)
if uploaded_file is not None:
    with open(uploaded_file.name, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    shutil.move(uploaded_file.name, os.path.join(os.getcwd(), 'pdfs'))
    st.session_state['pdf_file'] = uploaded_file.name
    with st.spinner('Extracting'):
        vb_list = vector_database_prep(uploaded_file)
    st.session_state['vb_list'] = vb_list
    os.chdir('pages')
    st.switch_page('rag.py')
