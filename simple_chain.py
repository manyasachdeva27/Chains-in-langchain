from langchain.chains import SimpleChain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()

#1 Component - Prompt Template
prompt = PromptTemplate(
    template="Give me a joke about {topic}",
    input_variables=["topic"]
)

#2 Component - Model
model=  GoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_GENAI_API_KEY"))

#3 Component - Output Parser
parser=StrOutputParser()

#4 Component - Chain
chain = prompt | model | parser

#5 Component - Run the chain
result = chain.invoke({"topic": "chickens"})

print(result)

chain.get_graph().print_ascii()