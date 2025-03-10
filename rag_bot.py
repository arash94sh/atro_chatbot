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
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages




if not os.environ.get("MISTRAL_API_KEY"):
  os.environ["MISTRAL_API_KEY"] = "9rgqfj9t0iqLe59uG7Q881iNXU6G0U5W"
  print(" key is set")
else:
    print (os.environ.get("MISTRAL_API_KEY"))
embeddings = MistralAIEmbeddings(model="mistral-embed", api_key="9rgqfj9t0iqLe59uG7Q881iNXU6G0U5W")
llm = init_chat_model("mistral-large-latest", model_provider="mistralai")


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",)  # Where to save data locally, remove if not necessary




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



    # Format into prompt

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "you must answer in farsi"
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type == 'human'

    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}





graph_builder.add_node(retrieve)
graph_builder.add_node(generate)

graph_builder.add_edge(START, "retrieve")

graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()












st.title("💬 Atro Agency Bot")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if input_text := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": input_text})
    st.chat_message("user").write(input_text)


    msg = graph.invoke({"messages":[HumanMessage(content=input_text)]})['messages'][-1].content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)



#____________________________________________________________