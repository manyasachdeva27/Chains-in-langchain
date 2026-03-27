from langchain.chains import SimpleChain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema.runnable import RunnableParallel
import os

load_dotenv()

model1 = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

model2 = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=os.getenv("CLAUDE_API_KEY"))

prompt1 = PromptTemplate(
    template="Give me short and simple notes about \n {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Give me 5 short question and answer about \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="merge the following notes and questions into a single document: \n {notes} \n {questions}",
    input_variables=["notes", "questions"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "questions": prompt2 | model2 | parser
}) 

merged_chain = prompt3 |  model1 | parser

chain = parallel_chain | merged_chain

result = chain.invoke({"topic": "AI"})

print(result)

chain.get_graph().print_ascii()