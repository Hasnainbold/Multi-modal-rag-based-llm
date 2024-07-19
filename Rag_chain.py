from langchain_openai import ChatOpenAI
import requests
from typing_extensions import TypedDict
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Weaviate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import *
from datasets import Dataset
import os
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, MessageGraph, Graph, StateGraph
from ragas import evaluate
import matplotlib.pyplot as plt
from langchain_core.runnables import RunnableLambda
from Query_agent import *
from Databases import *


class RAGEval:
    """
    Feedback-assisted agentic textual-visual RAG based LLM

    Utility method:
    1. Call RAGEval()
    2. Call model_prep()
    3. Call query_agent_prep()
    4. Call feedback_prep()
    5. Call imagedb_prep()
    6. Call query()
    """

    best = 2
    parse = StrOutputParser()

    def __init__(self, vb_list, cross_model):  # , q_model, q_parser, q_choice=1): # vb_list = [(vb,splitter)]
        self.cross_model = cross_model

        self.template = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Else, answer as a human being would.
        Use two sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.vb_list = vb_list

    def ground_truths_prep(self, questions):  # questions is a file with questions
        """
          DEPRECATED
        """

        self.ground_truths = [[s] for s in self.query(questions)]

    def model_prep(self, model, parser_choice=parse):  # model_link is the link to the model
        """
          Prepares the LLM and parser
        """

        self.chat_model = model
        self.parser = parser_choice

    def query_agent_prep(self, model, parser=parse):
        """
          Prepares the query agent
          Current Options:
          1. ReActQueryAgent
          2. AlternateQuestionAgent
          3. AugmentedQueryAgent
          4. TreeOfThoughtAgent
        """

        # self.query_agent = RunnableLambda(QueryAgent(self.vb_list, model,self.cross_model, parser).query)
        # self.query_agent = RunnableLambda(AlternateQuestionAgent(self.vb_list, model, self.cross_model, parser).query)
        self.query_agent = RunnableLambda(TreeOfThoughtAgent(self.vb_list, model, self.cross_model, parser).query)
        # self.query_agent = RunnableLambda(AugmentedQueryAgent(self.vb_list, model,self.cross_model,parser).query)

    def feedback_prep(self, uri, table_name, embedder, splitter, file):
        """
          Prepares the feedback retriever
          Current Options for the Database:
            1. Weaviate
            2. Pinecone
        """
        self.fd_db = TextDatabase(table_name, uri)
        self.fd_db.model_prep(embedder, splitter)
        with open(file) as f:
            data = f.read()
        self.fd_db.upsert(data)
        self.fd_db.retriever(top_k=5)

    def context_prep(self):
        """
          Internal Method for context preparation for a given question
        """

        con = self.query_agent.invoke(self.question).split('@@')
        uni_con = []
        for i in con:
            if i not in uni_con:
                uni_con.append(i)

        c = self.cross_model.rank(
            query=self.question,
            documents=uni_con,
            return_documents=True
        )[:self.best]
        self.context = str("\n".join([i['text'] for i in c]))

    def rag_graph(self):
        """
          Main Text-to-Text RAG graph method
          Utilises the following components:
          1. Feedback retriever
          2. Context Fetcher
          3. LLM
        """

        class GraphState(TypedDict):

            """
            Represents the state of our graph.

            Attributes:
                question: question
                context: context
                answer: answer
            """
            question: str
            context: str
            answer: str

        fd_retriever = self.fd_db.retriever(top_k=1)

        # state : question, context, answer

        def feedback(state):  # state modifier
            """
              Feedback Node Function
              Adds answer as feedback to the state
            """

            datas = fd_retriever.invoke(state["question"])
            data = datas[0]
            answer = data[data.find('and the response is') + len('and the response is'):]
            self.context = ""
            return {"question": state["question"], "context": self.context, "answer": answer}

        def feedback_check(state):  # state modifier
            """
              Feedback Checker Function
              If the feedback matches the query or not
            """

            datas = fd_retriever.invoke(state["question"])
            data = datas[0]
            q = data[len('The feedback for'):]
            q = q[:q.find('and the response is')].strip()
            q = ((" ".join(q.split(' ')[:-3])).lower()).strip()
            print(f'Feedback Question is {q}')
            if q == (state["question"].lower()).strip():
                return "f_answer"
            else:
                return "fetch"

        def fetch(state):  # state modifier
            """
              Fetch Node Function
              Adds context to the state
            """

            self.context_prep()
            return {"question": state["question"], "context": self.context, "answer": ""}

        def answer(state):  # state modifier
            """
              Answer Node Function
              Adds the answer to the state
            """

            chain = {"context": RunnableLambda(lambda x: state["context"]),
                     "question": RunnablePassthrough()} | self.prompt | self.chat_model | self.parser
            ans = chain.invoke(state["question"])
            return {"question": state["question"], "context": state["context"], "answer": ans}

        def feedback_answer(state):
            """
              DEPRECATED
            """

            template = """
          You are an assistant for question-answering tasks. You are given a question and its answer in a short form. Eloborate the answer till 2 sentences.
          Question: {question}
          Answer: {answer}
          """
            prompt = ChatPromptTemplate.from_template(template)
            chain = {"question": RunnablePassthrough(),
                     "answer": RunnableLambda(lambda x: state["answer"])} | prompt | self.chat_model | self.parser
            return {"question": state["question"], "context": state["context"],
                    "answer": chain.invoke(state["question"])}

        self.RAGraph = StateGraph(GraphState)
        self.RAGraph.set_entry_point("entry")
        self.RAGraph.add_node("entry", RunnablePassthrough())
        self.RAGraph.add_node("feedback", feedback)
        self.RAGraph.add_node("fetch", fetch)
        self.RAGraph.add_node("answerer", answer)
        # self.RAGraph.add_node("f_answer", feedback_answer)
        self.RAGraph.add_edge("entry", "feedback")
        self.RAGraph.add_conditional_edges(  # conditional edge based on feedback check
            "feedback",
            feedback_check,
            {"f_answer": END, "fetch": "fetch"}
        )
        # self.RAGraph.add_edge("f_answer", END)
        self.RAGraph.add_edge("fetch", "answerer")
        self.RAGraph.add_edge("answerer", END)
        self.ragchain = self.RAGraph.compile()

    def query(self, question, top_k=2):
        """
          Returns text and image results for a given question
        """

        if type(question) is str:  # if query is text
            print(f"MAIN QUESTION {question}")
            self.question = question
            state = {"question": self.question, "context": "", "answer": ""}
            self.rag_graph()
            answer_state = self.ragchain.invoke(state)
            self.answer = answer_state["answer"]
            text = self.answer
        else:  # query is an image
            text = self.image2text(question)  # get textual information of an image
            self.context = ""
        image = self.image_search(question, top_k)
        return {"text": text, "image": image, "context":self.context}

    def image_search(self, question, top_k=2):
        """
          Returns list of images associated with the query
        """

        result = [vb.query(question, top_k) for vb in self.vb_list]  # list[dic['image_data', 'text_data']]
        image_details = [i['image_data'] for i in result]  #
        images = []
        for i in image_details:
            for j in i['image']:
                images.append(j)
        unique_images = []
        for i in images:
            if i not in unique_images:
                unique_images.append(i)
        return unique_images  # list

    def image2text(self, question, top_k=2):
        result = [vb.query(question, top_k) for vb in self.vb_list] # list[dic['image_data', 'text_data']]
        image_details = [i['image_data'] for i in result] # list[dict[list, list]]
        return ["".join(i['context']) for i in image_details]

    def ragas(self, raise_exceptions=False):
        """
          Runs RAGAS evaluation on the RAG output
        """

        data = {
            "question": [self.question],
            "answer": [self.answer],
            "contexts": [[self.context]],
            "ground_truth": [self.ground_truth]
        }
        dataset = Dataset.from_dict(data)
        print(dataset['question'])
        print(dataset['answer'])
        print(dataset['contexts'])
        print(dataset['ground_truth'])
        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            ],
            raise_exceptions=raise_exceptions
        )
        df = result.to_pandas()
        return df

# If making a custom Parser with init and invoke function, define it as follows
# parser = RunnableLambda(Parser().invoke)
# pass parser into the model_prep function
