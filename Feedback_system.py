import pandas as pd
from langchain.docstore.document import Document
import weaviate
import weaviate.classes as wvc
from langchain_core.runnables import RunnableLambda
from weaviate.auth import AuthApiKey
from weaviate.classes.init import AdditionalConfig, Timeout
from langchain_weaviate.vectorstores import WeaviateVectorStore
from pinecone import Pinecone, ServerlessSpec
import uuid
from Databases import TextDatabase


class FeedbackSystem:
  """
    Takes a feedback from an existing file or the user and writes it to the vector database
    Queries the feedback for each question
    Vector Database support : Pinecone & Weaviate
  """

  top_k = 1
  def __init__(self,embeddings, api, v_type, **kwargs): # embeddings, api, v_type, url, index_name
    self.embeddings = embeddings
    self.v_type = v_type
    self.extra = {}
    for key,value in kwargs.items():
      self.extra[key] = value

    if self.v_type == 'Pinecone':
      self.vector_inst = Pinecone(api)
      index_name = self.extra['index_name']
      if index_name not in self.vector_inst.list_indexes().names():
        self.vector_inst.create_index(
          name=index_name,
          dimension=768,
          metric='cosine',
          spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
      self.index = self.vector_inst.Index(index_name)
    else:
      url = self.extra['url']
      self.vector_inst = weaviate.connect_to_wcs(
        cluster_url=url,
        auth_credentials=weaviate.auth.AuthApiKey(api),
        skip_init_checks=True,
        additional_config = AdditionalConfig(
          timeout=Timeout(init=300, query=300, insert=300)
        )
      )
      if 'Feedback' not in self.vector_inst.collections.list_all():
        self.vector_inst.collections.create(
          "Feedback",
          vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        )

  def feedback_retriever(self, top_k=1):
    """
      Settting up a retriever object
    """

    self.top_k = top_k
    self.retriever = RunnableLambda(self.fetch)

  def embed_feedback(self,feedback):
    """
      Embeds the feedback
    """

    embeds = self.embeddings.embed_query(feedback)
    return embeds

  def db_empty(self):
    """
      Checks if the database is empty
    """

    if self.v_type == 'Pinecone':
      return self.index.describe_index_stats()['total_vector_count'] == 0
    else:
      return len(self.vector_inst.collections.list_all()) == 0

  def write_file(self, feedback_file):
    """
      Reads a file and writes the feedbacks to the database only if database is empty
    """

    if self.db_empty():
      with open(feedback_file, 'r') as f:
        feedbacks = f.readlines()
      for fd in feedbacks:
        self.write(fd)

  def write(self, feedback):
    """
      Writes the feedback to the database
    """

    embed = self.embed_feedback(feedback)
    if self.v_type == 'Pinecone':
      question_objs = [{
        "id": str(uuid.uuid4()),
        "values": embed,
        "metadata": {"feedback": feedback}
      }]
      self.index.upsert(vectors=question_objs,namespace= "feedback")
    else:
      db_obj = [wvc.data.DataObject(properties={"feedback":feedback},vector=embed)]
      self.vector_inst.collections.get("Feedback").data.insert_many(db_obj)

  def fetch(self,question, top_k=5):
    """
      Fetches the feedback for the given question
    """

    q_embeds = self.embed_feedback(question)
    if self.v_type == 'Pinecone':
      responses = self.index.query(
        vector=q_embeds,
        top_k=top_k,
        namespace='feedback',
        include_metadata=True
      )
      return [Document(page_content=i['metadata']['feedback']) for i in responses.matches]
    else:
      response = self.vector_inst.collections.get("Feedback").query.near_vector(
        near_vector=q_embeds,
        limit=top_k,
        return_metadata=wvc.query.MetadataQuery(certainty=True)
      )
      return [Document(page_content=o.properties['feedback']) for o in response.objects]

  def clear_database(self):
    """
      Clears the database
    """

    if self.v_type == 'Pinecone':
      self.vector_inst.delete_index('feedback')
    else:
      self.vector_inst.collections.delete_all()
