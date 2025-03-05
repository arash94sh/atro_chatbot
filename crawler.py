from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
import requests
from bs4 import BeautifulSoup
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
import os

if not os.environ.get("MISTRALAI_API_KEY"):
  os.environ["MISTRALAI_API_KEY"] = "ct0DL5yDroEhR0PicB3wCn60TVkjre8Z"
else:
  print("Mistral AI API key already set in environment.")


os.environ['HF_TOKEN'] = "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"

from langchain_mistralai import MistralAIEmbeddings

embeddings = MistralAIEmbeddings(model="mistral-embed", api_key="ct0DL5yDroEhR0PicB3wCn60TVkjre8Z")

# Creating database
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
loader = WebBaseLoader(
    web_paths=("https://atro.agency/","https://atro.agency/services/webdesign/", "https://atro.agency/services/seo/", "https://atro.agency/services/gads/", "https://atro.agency/services/socialmedia/", "https://atro.agency/services/branding/"),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("page-content")
        )
    ),
)
docs = loader.load()
#chuncking and indexing the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

_ = vector_store.add_documents(documents=all_splits)





