from langchain_core.runnables import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import re


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

  def __init__(self,vb_list,agent, cross_model, parser=StrOutputParser()):
    self.prompt = ChatPromptTemplate.from_template(
      template="""You are given a question {question}.
      Based on it, you must only generate two alternate questions separated by newlines and numbered which are related to the given question.""",
    )
    self.model = agent
    self.parser = parser
    self.vb_list = vb_list
    self.cross_model = cross_model
    self.chain = {"question":RunnablePassthrough()} | self.prompt | self.model | self.parser

  def retrieve(self,question):
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
        )[:len(prior_context)-self.best+1]
    return [i['text'] for i in c] # list of text

  def query(self,question):
    qs = [i[3:] for i in (self.chain.invoke(question)).split('\n')] + [question]
    if '' in qs:
      qs.remove('')
    uni_q = []
    for q in qs:
      if q not in uni_q:
        uni_q.append(q)
    questions = [question]+uni_q # assuming the questions are labelled as 1. q1 \n 2. q2

    contexts = [self.retrieve(q) for q in questions]
    uni_contexts = []
    for i in contexts:
      for j in i:
        if j not in uni_contexts:
          uni_contexts.append(j)
    u = []
    for i in uni_contexts:
      k = re.split("(\.|\?|!)\n", i)
      for j in k:
        if j in '.?!':
          continue
        if j not in u:
          u.append(j)
    uni_contexts = []
    for i in range(len(u)):
      for j in range(len(u)):
        if j != i and u[i] in u[j]:
            break
      else:
        uni_contexts.append(u[i])
    # uni_contexts = u
    contexts = "@@".join(uni_contexts)
    return contexts


class SubQueryAgent:
    best = 2
    turns = 3

    class QueryGen:
        def __init__(self, q_model, parser=RunnableLambda(lambda x: x), prompt="""
        You will be given a pair of question and its context as an input.You must form a question contextually related to both of them.
        Question : {Question}\nContext: {Context}
        Output should in the format: sub-question : <sub_question>"""):
            self.context = ""
            self.prompt = ChatPromptTemplate.from_template(prompt.strip())
            self.chain = {"Question": RunnablePassthrough(),
                          "Context": RunnableLambda(lambda c: self.context)} | self.prompt | q_model | parser

        def __call__(self, question, context=""):
            self.context = context
            return self.chain.invoke(question)

    def __init__(self, vb_list, q_model, cross_model, parser=RunnableLambda(lambda x: x)):
        self.vb_list = vb_list
        self.q_model = q_model
        self.parser = parser
        self.cross_model = cross_model

    def fetch(self, question):
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
        return [i['text'] for i in c]  # list of text

    def query(self, question):
        agent = self.QueryGen(self.q_model, self.parser)
        sub_q = agent(question)
        print(f"Sub question: {sub_q}\n")

        contexts = []
        prompt = f"""You are given a main Question {question} and a pair of its subquestion and related sub context.
    You must generate a question based on the main question, and all of the sub-question and sub-contexts pairs.
    Output should in the format: sub-question : <sub_question>"""
        for i in range(self.turns - 1):
            print(f"ITERATION NO: {i}")
            context = self.fetch(sub_q)
            contexts += context
            prompt += "\nsub-question : {Question}\nsub-context: {Context}"
            agent = self.QueryGen(self.q_model, self.parser, prompt=prompt)
            sub_q = agent(sub_q, context)
            print(f"Sub question: {sub_q}\n")
        uni = []
        for c in contexts:
            if c not in uni:
                uni.append(c)
        return "@@".join(uni)


class TreeOfThoughtAgent:
    def __init__(self, vb_list, model, cross_model, parser=RunnableLambda(lambda x: x)):
        self.alt_agent = RunnableLambda(AlternateQuestionAgent(vb_list, model, cross_model, parser).mul_qs)
        self.sub_agent = RunnableLambda(SubQueryAgent(vb_list, model, cross_model, parser).query)

    def query(self, question):
        contexts = []
        mul_q = self.alt_agent.invoke(question)
        for q in mul_q:
            print(f"Question: {q}")
            sub_contexts = self.sub_agent.invoke(q)
            contexts.append(sub_contexts)
        return self.context_clean(contexts)

    def context_clean(self, contexts):
        uni_contexts = []
        for i in contexts:
            if i not in uni_contexts:
                uni_contexts.append(i)
        u = []
        for i in uni_contexts:
            k = re.split("(\.|\?|!)\n", i)
            for j in k:
                if j in '.?!':
                    continue
                if j not in u:
                    u.append(j)
        uni_contexts = []
        for i in range(len(u)):
            for j in range(len(u)):
                if j != i and u[i] in u[j]:
                    break
            else:
                uni_contexts.append(u[i])
        # uni_contexts = u
        return "@@".join(uni_contexts)
