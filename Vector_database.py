from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Weaviate
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
            index_name = self.extra['index']
            self.vector_inst = Pinecone(api_key=api_key)
            if index_name not in self.vector_inst.list_indexes().names():
                self.vector_inst.create_index(
                    name=index_name,
                    dimension=self.extra['dimension'],
                    metric=self.extra['metric'],
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            self.index = self.vector_inst.Index(index_name)
        else:
            self.vector_inst = weaviate.connect_to_wcs(
                cluster_url=self.extra['url'],
                auth_credentials=weaviate.classes.init.Auth.api_key(api_key)
            )

    def db_empty(self):
        if self.v_type == 'Pinecone':
            return self.index.describe_index_stats()['total_vector_count'] == 0
        else:
            return len(self.vector_inst.collections.list_all()) == 0

    def write(self, data_file, splitter):
        if self.db_empty():
            chunks = splitter.split_documents([Document(page_content=data_file)])
            if self.v_type == 'Pinecone':
                self.retriever = PineconeVectorStore.from_documents(
                    chunks,
                    self.embedding_model,
                    index_name=self.extra['index']
                ).as_retriever(search_type='similarity')
            else:
                self.retriever = WeaviateVectorStore.from_documents(
                    chunks,
                    self.embedding_model,
                    client=self.vector_inst
                ).as_retriever(search_type="mmr")

    def query(self, question):
        context = [doc.page_content for doc in self.retriever.invoke(question)]
        if self.v_type == 'Pinecone':
            print(f"Pinecone retrieved : {len(context)}")
        else:
            print(f"Weaviate retrieved : {len(context)}")

        c = self.cross_encoder.rank(
            query=question,
            documents=context,
            return_documents=True
        )[:len(context) - 2]
        return [i['text'] for i in c]

    def delete(self):
        if self.v_type == 'Pinecone':
            self.vector_inst.delete_index(self.extra['index'])
        else:
            self.vector_inst.collections.delete_all()


# vb = VectorDatabase(embedding_model, cross_encoder, 'Pinecone', pinecone_api_key, index='INDEX_NAME', dimension=DIMENSION_VALUE, metric='METRIC',url=None)
# vb = VectorDatabase(embedding_model, cross_encoder, 'Weaviate', WEAVIATE_API_KEY, index=None, dimension=None, metric=None, url=WEAVIATE_URL)
# vb.upsert(data,text_splitter1)
# context = vb.query(query1)
