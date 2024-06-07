from langchain_openai import ChatOpenAI
import requests
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Weaviate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import *
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
from ragas import evaluate
from langchain_core.runnables import RunnableLambda
from Vector_database import VectorDatabase


class RAGEval:
    """
    WorkFlow:
    1. Call RAGEval()
    2. Call ground_truth_prep()
    3. Call model_prep()
    4. Call query()
    5. Call raga()
    """

    best = 2
    parse = StrOutputParser()

    def __init__(self, file_path, vb_list, cross_model):  # vb_list = [(vb,splitter)]
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

    # def ground_truths_prep(self, questions):  # questions is a file with questions
    #     self.ground_truths = [[s] for s in self.query(questions)]
    #     self.vector_db(self.file_path, self.url, self.vb_key, self.embed_model)

    def vector_dbs(self):
        with open(self.file_path, 'r') as file:
            data = file.read()
        for vb, sp in self.vb_list:
            vb.upsert(data, sp)

    def model_prep(self, model, parser_choice=parse):  # model_link is the link to the model
        self.chat_model = model
        self.parser = parser_choice
        self.rag_chain()

    def rag_chain(self):
        def retrieve(question):
            prior_context = [vb.query(question) for vb, _ in self.vb_list]
            cont = []
            for i in prior_context:
                context = ""
                for j in i:  # list to str
                    context += j
                cont.append(context)

            c = self.cross_model.rank(
                query=question,
                documents=cont,
                return_documents=True
            )[:len(prior_context) - self.best + 1]
            self.context = [i['text'] for i in c]
            return self.context

        prompt = ChatPromptTemplate.from_template(self.template)
        self.retriever = RunnableLambda(retrieve)
        self.ragchain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | prompt
                | self.chat_model
                | self.parser
        )

    def query(self, question):
        self.questions = question
        self.answers = self.ragchain.invoke(question)
        return self.answers

    def raga(self):
        data = {
            "question": self.questions,
            "answer": self.answers,
            "contexts": self.context,
            "ground_truth": self.ground_truths
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            ]
        )
        df = result.to_pandas()
        return df

# If making a custom Parser with init and invoke function, define it as follows
# parser = RunnableLambda(Parser().invoke)
# pass parser into the model_prep function
