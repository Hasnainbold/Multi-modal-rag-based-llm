import torch
import weaviate.classes as wvc
from PIL import Image
import torchvision.transforms as T
from datasets import Dataset
import io
import weaviate
import pytesseract
from pinecone import Pinecone, ServerlessSpec
import uuid
import time
from weaviate.auth import AuthApiKey
from weaviate.classes.init import AdditionalConfig, Timeout
from abc import ABC, abstractmethod
import lancedb
from langchain_core.runnables import RunnableLambda


class Database(ABC):
  def __init__(self, table_name, uri='lancedb/rag'):
    self.db = lancedb.connect(uri)
    self.table_name = table_name
    try:
      self.delete()
    except:
      pass

  def upsert(self, data):
    self.tbl = self.db.create_table(self.table_name, data=data)

  def query(self, query_str, top_k=2):
    return self.tbl.search(query_str).limit(top_k).to_pandas()

  def delete(self):
    self.db.drop_table(self.table_name)

  def is_empty(self):
    return self.tbl.count_rows() == 0


class ImageDatabase(Database):
  top_k = 2

  def __init__(self, table_name, uri):
    self.im_db = Database(table_name + '_img', uri)
    self.txt_db = Database(table_name + '_txt', uri)

  def image_model_prep(self, extractor, model):
    self.extractor = extractor
    self.model = model
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.transformation_chain = T.Compose([
      T.Resize(int((256 / 224) * self.extractor.size["height"])),
      T.CenterCrop(self.extractor.size["height"]),
      T.ToTensor(),
      lambda x: x[:3, :, :] if x.shape[0] >= 3 else x.repeat(3, 1, 1),
      T.Normalize(mean=self.extractor.image_mean, std=self.extractor.image_std)
    ])

  def text_model_prep(self, embedder):
    self.embedder = embedder

  def _get_image_embedding(self, image):
    image_transformed = self.transformation_chain(image).unsqueeze(0)
    new_batch = {"pixel_values": image_transformed.to(self.device)}
    with torch.no_grad():
      embeddings = self.model(**new_batch).last_hidden_state[:, 0].cpu()
    return embeddings.flatten().tolist()

  def _get_text_embedding(self, text):
    return self.embedder.embed_query(text)

  def upsert(self, data):  # image_file, image_context, PIL Object
    if isinstance(data, list) and all(isinstance(i, tuple) and len(i) == 3 for i in data):
      image_embeddings = [{"image_file": i[0], "image_context": i[1], "vector": self._get_image_embedding(i[2])} for i
                          in data]
      text_embeddings = [{"image_file": i[0], "image_context": i[1], "vector": self._get_text_embedding(i[1])} for i in
                         data]
    elif isinstance(data, tuple) and len(data) == 3:
      image_embeddings = [
        {"image_file": data[0], "image_context": data[1], "vector": self._get_image_embedding(data[2])}]
      text_embeddings = [{"image_file": data[0], "image_context": data[1], "vector": self._get_text_embedding(data[1])}
                         for i in data]
    else:
      raise TypeError("Data should be a list of tuples or a single tuple")

    self.im_db.upsert(image_embeddings)
    self.txt_db.upsert(text_embeddings)

  def query(self, data, top_k=2):
    self.top_k = top_k
    if isinstance(data, Image.Image):  # image 2 image
      image_embedding = self._get_image_embedding(data)
      result = self.im_db.query(image_embedding, self.top_k)  # image + text
    elif isinstance(data, str):  # text 2 image
      text_embedding = self._get_text_embedding(data)
      result = self.txt_db.query(text_embedding, self.top_k)  # image + text
    else:
      raise TypeError('Data has to be a string or an PIL Image')
    return {"image": list(result['image_file']), "context": list(result['image_context'])}

  def delete(self):
    self.im_db.delete()
    self.txt_db.delete()

  def is_empty(self):
    return self.im_db.is_empty() and self.txt_db.is_empty()

  def retriever(self, top_k=2):
    self.top_k = top_k
    return RunnableLambda(self.query)


class TextDatabase(Database):
  top_k = 2

  def __init__(self, table_name, uri):
    super().__init__(table_name, uri)

  def model_prep(self, embedder, splitter):
    self.embedder = embedder
    self.splitter = splitter

  def upsert(self, data):  # data is str
    if isinstance(data, str):
      chunks = self.splitter.split_documents(self.splitter.create_documents(self.splitter.split_text(data)))
      chunks = [c.page_content for c in chunks]
      chunk_embeddings = [{"chunk":chunk,"vector":self.embedder.embed_documents(chunk)} for chunk in chunks]
      super().upsert(chunk_embeddings)
      #self.tbl.create_index(num_partitions=256, num_sub_vectors=96)
    else:
      raise TypeError("Data should be a string")

  def query(self, data, top_k=2): # str
    self.top_k = top_k
    embedding = self.embedder.embed_query(data)
    return super().query(embedding, self.top_k)['chunk']  # text

  def retriever(self, top_k):
    self.top_k = top_k
    return RunnableLambda(self.query)


class UnifiedDatabase(ImageDatabase, TextDatabase):  # Assuming TextDatabase is defined similarly
  def __init__(self, table_name, uri):
    self.im_table_name = table_name + '_image'
    self.txt_table_name = table_name + '_text'
    ImageDatabase.__init__(self, self.im_table_name, uri)
    TextDatabase.__init__(self, self.txt_table_name, uri)  # Uncomment if TextDatabase is defined

  def model_prep(self, extractor, model, embedder, splitter):
    ImageDatabase.image_model_prep(self, extractor, model)
    ImageDatabase.text_model_prep(self, embedder)
    TextDatabase.model_prep(self, embedder, splitter)  # Uncomment if TextDatabase is defined

  def upsert(self, data):
    if isinstance(data, str):  # text
      TextDatabase.upsert(self, data)  # Uncomment if TextDatabase is defined
    elif isinstance(data, list) and isinstance(data[0], tuple):  # image
      ImageDatabase.upsert(self, data)

  def query(self, data, top_k=2):  # image, text
    if isinstance(data, str):  # text
      image_data = ImageDatabase.query(self, data, top_k)  # image, text
      text_data = TextDatabase.query(self, data, top_k)  # text
      return {"image_data": image_data, "text_data": list(text_data)}
    elif isinstance(data, Image.Image):  # image
      image_data = ImageDatabase.query(self, data, top_k)  # dict[list, list]
      return {"image_data": image_data, "text_data": []}
    else:
      raise TypeError('Data has to be a string or an PIL Image')

  def delete(self):
    ImageDatabase.delete(self)
    TextDatabase.delete(self)  # Uncomment if TextDatabase is defined

  def is_empty(self):
    return ImageDatabase.is_empty(self) and TextDatabase.is_empty(self)  # Uncomment if TextDatabase is defined
