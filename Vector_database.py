from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate


class VectorDatabase:
  def __init__(self, embedding_model, cross_encoder, v_type, api_key, **kwargs):  # kwargs : index, dimension, metric, url
    self.embedding_model = embedding_model
    self.cross_encoder = cross_encoder
    self.v_type = v_type
    self.extra = {}
    for key, value in kwargs.items():
      self.extra[key] = value

    if self.v_type == 'Pinecone':
      self.vector_inst = Pinecone(api_key=api_key)
      index_name = self.extra['index']
      existing_indexes = [index_info["name"] for index_info in self.vector_inst.list_indexes()]

      if index_name not in existing_indexes:
          self.vector_inst.create_index(
              name=index_name,
              dimension=self.extra['dimension'],
              metric=self.extra['metric'],
              spec=ServerlessSpec(cloud="aws", region="us-east-1"),
          )
    else:
      self.vector_inst = weaviate.connect_to_wcs(cluster_url=self.extra['url'], auth_credentials=weaviate.classes.init.Auth.api_key(api_key))

  def data_prep(self, data, splitter):
    documents = [Document(page_content=data)]
    text_splitter = splitter
    chunks = text_splitter.split_documents(documents)
    return chunks

  def upsert(self, data, splitter):
    chunks = self.data_prep(data, splitter)
    if self.v_type == 'Pinecone':
      self.retriever = PineconeVectorStore.from_documents(chunks, self.embedding_model, index_name=self.extra['index'])
    else:
      db = WeaviateVectorStore.from_documents(chunks, self.embedding_model, client=self.vector_inst)
      self.retriever = db.as_retriever(search_type="mmr")

  def query(self, question):
      if self.v_type == 'Pinecone':
          context = [doc.page_content for doc in self.retriever.similarity_search(question)]
      else:
          context = [doc.page_content for doc in self.retriever.invoke(question)]
      c = self.cross_encoder.rank(
          query=question,
          documents=context,
          return_documents=True
      )[:len(context) - 2]
      return [i['text'] for i in c]


# vb = VectorDatabase(embedding_model, cross_encoder, 'Pinecone', pinecone_api_key, index='INDEX_NAME', dimension=DIMENSION_VALUE, metric='METRIC',url=None)
# vb = VectorDatabase(embedding_model, cross_encoder, 'Weaviate', WEAVIATE_API_KEY, index=None, dimension=None, metric=None, url=WEAVIATE_URL)
# vb.upsert(data,text_splitter1)
# context = vb.query(query1)
