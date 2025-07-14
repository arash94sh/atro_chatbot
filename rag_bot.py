import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import bs4
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_mistralai import ChatMistralAI
from langchain.chat_models import init_chat_model
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph, MessagesState
from typing_extensions import List, TypedDict
import os
from langchain_mistralai import MistralAIEmbeddings
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage  
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI




if not os.environ.get("MISTRAL_API_KEY"):
  os.environ["MISTRAL_API_KEY"] = "9rgqfj9t0iqLe59uG7Q881iNXU6G0U5W"
  print(" key is set")
else:
    print (os.environ.get("MISTRAL_API_KEY"))
embeddings = MistralAIEmbeddings(model="mistral-embed", api_key="9rgqfj9t0iqLe59uG7Q881iNXU6G0U5W")
#llm = init_chat_model("mistral-large-latest", model_provider="mistralai")


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",)  # Where to save data locally, remove if not necessary

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    max_retries=2,
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySUQiOiI2N2ZlMGJkNmIwMmU2Y2Q5YzY3NTJlMzQiLCJ0eXBlIjoiYXV0aCIsImlhdCI6MTc1MTI3MTgwNn0.Aw2FYlyAdqvhb5qOAj4K3xG1RAyaO1GgIhJFsewM1p4",
    base_url="https://ai.liara.ir/api/v1/6860e58052ae45201e7cdd38",
    timeout = 20.0,
    temperature=0,
    # organization="...",
    # other params...
)


class State(TypedDict):
  
  messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["messages"][0].content)
    content = "\n\n".join([doc.page_content for doc in retrieved_docs]) 
    return {"messages":[SystemMessage(content)]}











# Step 3: Generate a response using the retrieved content.
def generate(state: State):
    """Generate answer."""
    # Get generated ToolMessages
    docs_content = "" 
    for message in state["messages"]:
        if message.type == "system":
            docs_content = message.content
        if docs_content == "":
            docs_content = " there is no content so just say you don't know" 




    # Format into prompt

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "you must answer in farsi"
        "\n\n"
        "retrieved content:"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type == 'human'

    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    # Run
    response = llm.with_retry().invoke(prompt)
    return {"messages": [response]}





graph_builder.add_node(retrieve)
graph_builder.add_node(generate)

graph_builder.add_edge(START, "retrieve")

graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)












st.title("💬 Atro Agency Bot")

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "سلام٬به پشتیبان وبسایت آترو خوش آمدید. لطفا سوال خود را بپرسید"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if input_text := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": input_text})
    st.chat_message("user").write(input_text)

    config = {"configurable": {"thread_id": st.session_state.session_id}}
    msg = graph.invoke({"messages":[HumanMessage(content=input_text)]}, config)['messages'][-1].content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)



#____________________________________________________________