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
from Query_agent import *


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

    def __init__(self, vb_list, cross_model):  # vb_list = [(vb,splitter)]
        self.cross_model = cross_model

        self.template = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use two sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:
        """
        self.vb_list = vb_list

    def ground_truths_prep(self, ground_truth):  # questions is a file with questions
        self.ground_truth = ground_truth

    def model_prep(self, model, parser_choice=parse):  # model_link is the link to the model
        self.chat_model = model
        self.parser = parser_choice

    def query_agent_prep(self,model,parser):
        self.query_agent = RunnableLambda(AlternateQuestionAgent(self.vb_list, model, self.cross_model, parser).query)

    def context_prep(self):
        con = self.query_agent.invoke(self.question).split('@@')
        uni_con = []
        for i in con:
            if i not in uni_con:
                uni_con.append(i)
        self.context = str("\n".join(uni_con))

    def rag_chain(self):
        prompt = ChatPromptTemplate.from_template(self.template)
        self.context_prep()
        context_agent = RunnableLambda(lambda x: str(self.context))
        self.ragchain = (
            {"context": context_agent, "question": RunnablePassthrough()}
                  | prompt
                  | self.chat_model
                  | self.parser
        )

    def query(self, question):
        self.question = question
        self.rag_chain()
        self.answer = self.ragchain.invoke(question)
        return self.answer

    def ragas(self):
        # self.context = [[c] for c in self.context]
        data = {
            "question": [self.question],
            "answer": [self.answer],
            "contexts": [[self.context]],
            "ground_truth": [self.ground_truth]
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
        return result.to_pandas()

# If making a custom Parser with init and invoke function, define it as follows
# parser = RunnableLambda(Parser().invoke)
# pass parser into the model_prep function
