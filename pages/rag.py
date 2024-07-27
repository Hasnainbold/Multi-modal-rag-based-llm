import re
from streamlit_feedback import streamlit_feedback
import streamlit as st
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)
from uuid import uuid4
from llama_index.core import Settings
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import *
from langsmith import Client
from openai import OpenAI
import base64
from io import BytesIO
import requests
import matplotlib.pyplot as plt
os.chdir('..')
from src.Rag_chain import *
from src.Query_agent import *
from langsmith.run_trees import RunTree
from src.Databases import *
os.chdir('pages')
showWarningOnDirectExecution = False


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

  def __init__(self, stopword='Answer:'):
    """
      Initiliases a StrOutputParser as the base parser
    """
    self.parser = StrOutputParser()
    self.stopword = stopword

  def invoke(self, query):
    """
      Invokes the parser and finds the Model response
    """
    ans = self.parser.invoke(query)
    return ans[ans.find(self.stopword)+len(self.stopword):].strip()


class ChatGPT:
  """
    Wrapper Class for ChatGPT Class
  """

  def __init__(self, model, api_key, template):
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


Settings.embed_model = st.session_state['Settings.embed_model']
processor, vision_model = st.session_state['processor'], st.session_state['vision_model']
bi_encoder = st.session_state['bi_encoder']
chat_model = st.session_state['chat_model']
cross_model = st.session_state['cross_model']
extractor, image_model = st.session_state['extractor'], st.session_state['image_model']
pinecone_embed = st.session_state['pinecone_embed']
weaviate_embed = st.session_state['weaviate_embed']

pdf_file = st.session_state['pdf_file']
file_name = pdf_file
image_folder = f'../figures_{file_name}'
url = st.secrets["WEAVIATE_URL"]
v_key = st.secrets["WEAVIATE_V_KEY"]
gpt_key = st.secrets["GPT_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["OPENAI_API_KEY"] = gpt_key
f_url = st.secrets['FEEDBACK_URL']
f_api = st.secrets['FEEDBACK_API']
feedback_file = "../feedback_loop.txt"
im_db_url = st.secrets["IMAGE_URL"]
im_db_key = st.secrets["IMAGE_API"]
txt_db_url = st.secrets["TEXT_URL"]
txt_db_key = st.secrets["TEXT_API"]

fd = False
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
mistral_parser = RunnableLambda(MistralParser().invoke)
vb_list = st.session_state['vb_list']
q_model = st.session_state['q_model']
alt_parser = RunnableLambda(MistralParser('alternate-questions :\n').invoke)
sub_parser = RunnableLambda(MistralParser('sub-question : ').invoke)
image_parser = RunnableLambda(MistralParser().invoke)
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
pine_embed = st.session_state['pinecone_embed']
feedback_db = TextDatabase('feedback', '../lancedb/rag')
feedback_db.model_prep(weaviate_embed, RecursiveCharacterTextSplitter(chunk_size=1330, chunk_overlap=35))
with open('../feedback_loop.txt', 'r') as f:
  feedback = f.read()
feedback_db.upsert(feedback)

req = RAGEval(vb_list, cross_model)
req.model_prep(chat_model, mistral_parser)
req.query_agent_prep(q_model, (alt_parser, sub_parser, image_parser))
req.feedback_prep(uri='../lancedb/rag', table_name='feedback',
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

st.title('Multi-modal RAG based LLM for Information Retrieval')
st.subheader('Converse with our Chatbot')
st.markdown("You may input text or image")
st.markdown("Some sample questions to ask:")
st.markdown("- What are adjustment points in the context of using a microscope, and why are they important?")
st.markdown("- What does alignment accuracy refer to, and how is it achieved in a microscopy context?")
st.markdown("- What are alignment marks, and how are they used in the alignment process?")
st.markdown("- What is the alignment process in lithography, and how does eLitho facilitate this procedure?")
st.markdown("- What can you do with the insertable layer in Smart FIB?")

open('../feedback.txt', 'w').close()

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
    with open('../feedback.txt', 'r+') as fd:  # feedback records all feedback for this run
        fd.write(s)
    with open('../feedback_loop.txt', 'r+') as fd:  # feedback loop records feedback for all runs
        fd.write(s)
    feedback_db.upsert(s)
    with open('../feedback.txt', 'r') as fd:
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
    up_image = Image.open(uploaded_file)
    name = uploaded_file.name.split('/')[-1]
    up_image.save(name)

    associated_text = st.text_area("Provide a query wrt to image if any else input \'None\' ")
    while True:
        if len(associated_text):
            result = req.query(up_image, 5)  # dict
            images = [name] + result['image']  # list
            image_context = "Context for the image:\n" + "".join(result['text'])  # str

            if associated_text.lower().strip() != 'none':
                prompt = associated_text
                st.session_state.messages.append({"role": "user", "content": prompt})
                ai_response = req.query("Given " + image_context + '; Here is the user query: ' + prompt)['text']
                user_response = {"role": "user", "content": prompt}
            else:
                prompt = ""
                ai_response = image_context
                user_response = {}

            conv_id = uuid.uuid4()
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.session_state['image'] += [name] + images
            st.session_state['conv_id'][conv_id] = {
                "user_messages": user_response,
                "ai_messages": {"role": "assistant", "content": ai_response},
                "images": images  # list
            }
            break

if prompt := st.chat_input("What's Up?"):
    prompt = prompt
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
            st.markdown(user_messages["content"])
    with st.chat_message(ai_messages["role"]):
        st.markdown(ai_messages["content"])
    if len(images) > 0:
        for image in images:
            st.image(Image.open(os.path.join(image_folder, image)), use_column_width=True)

if fd:
    with st.form('form'):
        streamlit_feedback(feedback_type="thumbs", align="flex-start",
                           key='fb_k', optional_text_label="[Optional] Please provide an explanation")
        submit_button = st.form_submit_button('Save feedback', on_click=fbcb)
        if not submit_button:
            print('Click the Submit button')

with open('../feedback.txt', 'r')as f:
    rt.end(outputs={'outputs': f.read()})
rt.post()


def reset_conversation():
    st.session_state.messages = []
    st.session_state['image'] = []
    st.session_state['conv_id'] = {}


st.button('Reset Chat', on_click=reset_conversation)
