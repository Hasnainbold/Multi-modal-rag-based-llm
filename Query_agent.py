from langchain_core.runnables import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough


class QueryAgent:
    max_turns = 3
    best = 2
    prompt = """
        You will be given a pair of question and its context as an input.
        You must form a question contextually related to both of them.
        Format for input:
        Question : <Question>
        Context: <Context>

        Format for output:
        Output: <Output>
        """.strip()

    def __init__(self, vb_list, q_model, cross_model, parser=RunnableLambda(lambda x: x)):
        self.vb_list = vb_list
        self.q_model = q_model
        self.cross_model = cross_model
        self.parser = parser
        self.messages = [{"role": "system", "content": self.prompt}]

    def __call__(self, query, context):
        message = f"Question: {query}\nContext: {context}"
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def fetch(self, question):
        prior_context = [vb.query(question) for vb, _ in self.vb_list]
        cont = ["".join(i) for i in prior_context]
        c = self.cross_model.rank(
            query=question,
            documents=cont,
            return_documents=True
          )[:len(cont)-self.best+1]
        return [i['text'] for i in c]

    def execute(self):
        content = "\n".join([message["content"] for message in self.messages if (message["role"] != "assistant")])
        return self.parser.invoke(self.q_model.invoke(content, max_length=128, num_return_sequences=1))

    def query(self, question):
        self.question = question
        self.context, context = "", ""

        for i in range(self.max_turns):
            self.context += context + '\n'
            subq = self(question, context)
            print(f"Sub question: {subq}\n")
            question, context = subq, "".join(self.fetch(subq))
            print(f"Context: {context}\n")
        return self.context


class AlternateQuestionAgent:
  best = 2

  def __init__(self, vb_list, agent, cross_model, parser=StrOutputParser()):
      self.prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are a helpful assistant.
        Your task is to generate two different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by newlines and numbered.
        Original question: {question}""",
      )
      self.model = agent
      self.parser = parser
      self.vb_list = vb_list
      self.cross_model = cross_model
      self.chain = {"question": RunnablePassthrough()} | self.prompt | self.model | self.parser

  def retrieve(self, question):
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
          )[:len(prior_context)-self.best+1]
      return [i['text'] for i in c]  # list of text

  def query(self, question):
      qs = [i[3:] for i in (self.chain.invoke(question)).split('\n')]
      if '' in qs:
          qs.remove('')
      questions = [question] + qs  # assuming the questions are labelled as 1. q1 \n 2. q2
      context = ""
      for q in questions:
          context += "".join(self.retrieve(q)) + "@@"
      return context
