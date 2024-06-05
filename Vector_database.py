from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate


class VectorDatabase:
    def __init__(self, embedding_model, v_type, api_key, **kwargs):  # kwargs : index, dimension, metric, url
        self.embedding_model = embedding_model
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
            self.vector_inst = weaviate.connect_to_wcs(cluster_url=self.extra['url'],
                                                       auth_credentials=weaviate.classes.init.Auth.api_key(api_key))

    def data_prep(self, data, splitter):
        documents = [Document(page_content=data)]
        text_splitter = splitter
        chunks = text_splitter.split_documents(documents)
        return chunks

    def upsert(self, data, splitter):
        chunks = self.data_prep(data, splitter)
        if self.v_type == 'Pinecone':
            self.retriever = PineconeVectorStore.from_documents(chunks, self.embedding_model,
                                                                index_name=self.extra['index'])
        else:
            db = WeaviateVectorStore.from_documents(chunks, self.embedding_model, client=self.vector_inst)
            self.retriever = db.as_retriever(search_type="mmr")

    def query(self, question):
        if self.v_type == 'Pinecone':
            docsearch = self.retriever.similarity_search(question)
            context = [doc.page_content for doc in docsearch]
        else:
            matched_docs = self.retriever.invoke(question)
            context = [d.page_content for d in matched_docs]
        return context

# vb = VectorDatabase(embedding_model, 'Pinecone', pinecone_api_key, index='INDEX_NAME', dimension=DIMENSION_VALUE, metric='METRIC',url=None)
# vb = VectorDatabase(embedding_model, 'Weaviate', WEAVIATE_API_KEY, index=None, dimension=None, metric=None, url=WEAVIATE_URL)
# vb.upsert(data,text_splitter1)
# context = vb.query(query1)
