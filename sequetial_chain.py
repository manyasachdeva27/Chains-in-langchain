from langchain.chains import SimpleChain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()

#1 Component - Prompt Template
prompt1 = PromptTemplate(
    template="Give me a detailed report about {topic}",
    input_variables=["topic"]
)

#2 Component - Prompt Template
prompt2 = PromptTemplate(
    template="Give me a summary of the report {report}",
    input_variables=["report"]
)

#3 Component - Model
model=  GoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_GENAI_API_KEY"))

#4 Component - Output Parser
parser=StrOutputParser()

#5 Component - Chain
chain = prompt1 | model | parser | prompt2 | model | parser

#6 Component - Run the chain
result = chain.invoke({"topic": "AI"})

print(result)

chain.get_graph().print_ascii()