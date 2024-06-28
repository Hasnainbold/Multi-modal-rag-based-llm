import pandas as pd
from langchain.docstore.document import Document
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore


class FeedbackSystem:
    top_k = 1

    def __init__(self, file_path, embeddings, url, key):
        self.embeddings = embeddings
        self.df = pd.read_csv(file_path)
        feedback_vb = weaviate.connect_to_wcs(cluster_url=url,auth_credentials=weaviate.classes.init.Auth.api_key(key))
        self.db = WeaviateVectorStore.from_documents([Document(row['Text']) for _, row in self.df.iterrows()],
                                                     embeddings, client=feedback_vb)

    def feedback_retriever(self, top_k=1):
        self.top_k = top_k
        self.retriever = self.db.as_retriever(search_type="mmr", search_kwargs={'k': self.top_k})

    def fetch(self, question, top_k=1):
        self.feedback_retriever(top_k)
        data = self.retriever.invoke(question)
        return [i.page_content for i in data][:top_k]

    def write(self, feedback):
        self.db.add_documents([Document(feedback)])
        self.feedback_retriever(self.top_k)
