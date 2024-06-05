from langchain_openai import ChatOpenAI
import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
import os
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.llms import HuggingFaceHub
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
import streamlit as st
from sentence_transformers import SentenceTransformer
from ragas.metrics import faithfulness, answer_correctness, answer_similarity
from ragas import evaluate
from typing import Sequence
import pandas as pd
from Vector_database import VectorDatabase

class RAGEval:
    '''
    WorkFlow:
    1. Call RAGEval()
    2. Call ground_truth_prep()
    3. Call model_prep()
    4. Call query()
    5. Call raga()
    '''
    best = 2
    parse = StrOutputParser()

    def __init__(self,file_path, vb_list, cross_model): # vb_list = [(vb,splitter)]
        self.cross_model = cross_model
        self.template = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use two sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:
        """
        self.file_path = file_path
        self.vb_list = vb_list
        self.vector_dbs()

    def ground_truths_prep(self,questions): # questions is a file with questions
        self.ground_truths = [[s] for s in self.query(questions)]
        self.vector_db(self.file_path, self.url, self.vb_key, self.embed_model)

    def vector_dbs(self):
      with open(self.file_path, 'r') as file:
        data = file.read()
      for vb,sp in self.vb_list:
        vb.upsert(data,sp)

    def model_prep(self,model,parser_choice=parse): # model_link is the link to the model
        self.chat_model = model
        self.parser = parser_choice

    def query(self,question):
        self.questions = question
        prior_context = [vb.query(question) for vb,_ in self.vb_list]
        cont = []
        for i in prior_context:
          context = ""
          for j in i: # list to str
            context += j
          cont.append(context)

        c = self.cross_model.rank(
              query=question,
              documents=cont,
              return_documents=True
            )[:len(prior_context)-self.best]
        self.context = [i['text'] for i in c]

        self.answers = self.parser.invoke(
                self.chat_model.invoke(
                  self.template.format(
                    question=self.questions,
                    context=self.context)
                )
            )

        return self.answers

    def raga(self): # metric: 1 for Context_Precision / 2 for Context_Recall / 3 for Faithfulness / 4 for Answer_Relevancy
        data = {
            "question": self.questions,
            "answer": self.answers,
            "contexts": self.context,
            "ground_truth": self.ground_truths
        }
        dataset=Dataset.from_dict(data)
        result=evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            ]
        )
        df=result.to_pandas()
        return df
