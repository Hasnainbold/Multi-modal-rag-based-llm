from typing_extensions import TypedDict
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from langgraph.graph import END, StateGraph
from ragas import evaluate
from src.Query_agent import *
from src.Databases import *


class RAGEval:
    """
    Multi-modal RAG based LLM for Information Retrieval

    Utility method:
    1. Call RAGEval()
    2. Call model_prep()
    3. Call query_agent_prep()
    4. Call feedback_prep()
    5. Call imagedb_prep()
    6. Call query()
    """

    best = 4
    parse = StrOutputParser()

    def __init__(self, vb_list, cross_model):
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
        self.ground_truth = ""

    def ground_truths_prep(self, questions):  # questions is a file with questions
        """
          DEPRECATED
        """

        self.ground_truths = [[s] for s in self.query(questions)]

    def model_prep(self, model, parser_choice=parse):
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
          5. ImageContextAgent
        """

        # self.query_agent = RunnableLambda(QueryAgent(self.vb_list, model,self.cross_model, parser).query)
        # self.query_agent = RunnableLambda(AlternateQuestionAgent(self.vb_list, model, self.cross_model, parser).query)
        self.query_agent = RunnableLambda(TreeOfThoughtAgent(self.vb_list, model, self.cross_model, parser[:2]).query)
        self.context_agent = RunnableLambda(ImageContextAgent(model, parser[2]).reword)
        # self.query_agent = RunnableLambda(AugmentedQueryAgent(self.vb_list, model,self.cross_model,parser).query)

    def feedback_prep(self, uri, table_name, embedder, splitter, file):
        """
          Prepares the feedback retriever
          Current Options for the Database:
            1. Lancedb
        """
        self.fd_db = TextDatabase(table_name, uri)
        self.fd_db.model_prep(embedder, splitter)
        with open(file) as f:
            data = f.read()
        self.fd_db.upsert(data)
        self.fd_db.retriever(top_k=5)

    def _context_prep(self):
        """
          Internal Method for context preparation for a given question
        """

        def findWholeWord(w):
            return re.compile(r'\b{0}\b'.format(re.escape(w)), flags=re.IGNORECASE).search

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
        cons = [i['text'] for i in c]

        self.figure_mentions = []
        for c in cons:
            print('HERE')
            print(c)
            if findWholeWord('fig')(c) or findWholeWord('figure')(c):
                print('GOGO')
                self.figure_mentions.append(c.split(':')[0])

        self.context = str("\n".join(cons))

    def _rag_graph(self):
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

            self._context_prep()
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

        self.RAGraph = StateGraph(GraphState)
        self.RAGraph.set_entry_point("entry")
        self.RAGraph.add_node("entry", RunnablePassthrough())
        self.RAGraph.add_node("feedback", feedback)
        self.RAGraph.add_node("fetch", fetch)
        self.RAGraph.add_node("answerer", answer)
        self.RAGraph.add_edge("entry", "feedback")
        self.RAGraph.add_conditional_edges(  # conditional edge based on feedback check
            "feedback",
            feedback_check,
            {"f_answer": END, "fetch": "fetch"}
        )
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
            self._rag_graph()
            answer_state = self.ragchain.invoke(state)
            self.answer = answer_state["answer"]
            text = self.answer

            image = []
            if len(self.figure_mentions) != 0:
                print('FOUND ONE')
                for fig in self.figure_mentions:
                    for vb in self.vb_list:
                        k = vb.search_name(fig)
                        image += k['image_file'].tolist()
                return {"text": text, "image": image, "context": self.context}

        else:  # query is an image
            text = self._image2text(question)  # get textual information of an image
            self.context = ""
        image = self._image_search(question, top_k)
        return {"text": text, "image": image, "context": self.context}

    def _image_search(self, question, top_k=2):
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

    def _image2text(self, question, top_k=2):
        def unique(l):
            u = []
            for i in l:
                if i not in u:
                    u.append(i)
            return u

        result = [vb.query(question, top_k) for vb in self.vb_list]  # list[dic['image_data', 'text_data']]
        image_details = [i['image_data'] for i in result]  # list[dict[list, list]]
        contexts = unique(["\n".join(i['context']) for i in image_details])
        contexts = self.context_agent.invoke("\n".join(contexts))
        return contexts

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
