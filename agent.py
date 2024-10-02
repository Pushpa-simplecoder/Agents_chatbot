from dotenv import load_dotenv
import os
load_dotenv()
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
os.environ["Together_API_KEY"]=os.getenv("Together_API_KEY")
os.environ["USER_AGENT"]="myagent"
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
search=TavilySearchResults(max_results=3)
loader=WebBaseLoader("https://docs.smith.langchain.com/overview")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200).split_documents(docs)
model_name = "sentence-transformers/all-mpnet-base-v2"
hf=HuggingFaceEmbeddings(
    model_name=model_name
)
vectordb=FAISS.from_documents(documents,hf)
retriver=vectordb.as_retriever()
retriever_tool=create_retriever_tool(retriver,
                                     name="retriever_data",
                                     description="You can get any information about langchains"
                                     )
tools=[search,retriever_tool]
chat=ChatTogether(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
)
model_with_tools=chat.bind_tools(tools)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent=create_tool_calling_agent(chat,tools,prompt)
agent_excecutor=AgentExecutor(agent=agent,tools=tools)
questions=[]
answers=[]
file_name="POC.csv"
if os.path.exists(file_name):
    df_old=pd.read_csv("POC.csv")
else:
    df_new=pd.DataFrame(columns=["question","answer"])
while True:
    user_input=input("Ask your question")
    if user_input.lower()=="stop":
        print("Conversation ended")
        break
    questions.append(user_input)
    res=agent_excecutor.invoke({"input":user_input})
    answers.append(res)
    print(res)
df_new=pd.DataFrame({"question":questions,"answer":answers})
df_combined=pd.concat([df_old,df_new],ignore_index=False)
df_combined.to_csv(file_name,index=False)






