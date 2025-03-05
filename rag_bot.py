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
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import getpass
import os
from langchain_mistralai import MistralAIEmbeddings

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import os


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




#rpompt crafting
prompt = ChatPromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template="you are an assistant that answers the users question regarding the services a company provide based on the content of this website, this is the persian website and the users are persians so you must speak in farsi.\nQuestion: {question} \nContext: {context} \nAnswer:"), additional_kwargs={})])

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()





st.title("💬 Atro Agency Bot")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if input_text := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": input_text})
    st.chat_message("user").write(input_text)


    msg = graph.invoke({"question": f"{input_text}"})['answer']
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
