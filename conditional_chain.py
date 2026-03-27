from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema.runnable import RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.chains import ConditionalChain
from langchain.schema.runnable import RunnableParallel, RunnableBranch
from typing import Literal
import os

load_dotenv()

model1 = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback into positive or negative: \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="Give me a appropriate response for the positive feedback: \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Give me a appropriate response for the negative feedback: \n {feedback}",
    input_variables=["feedback"]
)

classify_chain = prompt1 | model1 | parser2

branch = RunnableBranch(

    (lambda x: x.sentiment == "positive", prompt2 | model1 | parser1),
    (lambda x: x.sentiment == "negative", prompt3 | model1 | parser1)
    default=lambda x: "Thank you for your feedback"
)

conditional_chain = classify_chain | branch