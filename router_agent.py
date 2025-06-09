from groq import Groq
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda

from dotenv import load_dotenv

load_dotenv()


mixtral = 'mixtral-8x7b-32768'
llama = 'llama3-70b-8192'


#model = ChatGroq(temperature=0, model_name=llama) 

model = ChatAnthropic(temperature=0.4, model_name='claude-3-5-sonnet-20240620') 

template = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent router in text to sql system that understands the user question and 
determines which agents might have answer to the question based on agent description. Multiple agents might answer a given user question. OUTPUT SHOULD BE IN FORM OF LIST OF strings.
Dont give any explanation or any other verbose in the output.
"""),

    ("human", '''
Below are descriptions of different agents.
customer agent : It contains all the details about customer and seller locations and their unique identifiers
orders agent : It contains details about all the orders like product identifier, order identifier, products in an order, no. of items of a product in order, price of order, frieght value, order time, delivery status and its time, payment etc.
product agent : It contains details about product like product identifier, product category, description, dimensions of product


STEP BY STEP TABLE SELECTION PROCESS:
- Split the question into different subquestions.
- For each subquestion, very carefully go through each and every AGENT description, think which agent might have answer to this subquestion.
- At the end collect all the agents that you thought can answer the whole question in form of list of strings
- For a give question, if customer and orders agents can answer question, give output like below without any verbose.
['customer', 'orders']
- If only customer can answer a question , give output like below with one table in list
['customer']
- For a give question, if customer and orders and product agent can answer question, give output like below without any verbose.
['customer', 'orders', 'product']
     
User question:
{question}

     ''')
])

# Fix the RunnableMap implementation
chain = (
    RunnableMap({
        "question": lambda x: x["question"]
    })
    | template 
    | model 
    | StrOutputParser()
)

def agent_2(q):
    response = chain.invoke({"question": q}).replace('```', '')
    return response


