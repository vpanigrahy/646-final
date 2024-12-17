from flask import Flask, request, jsonify, render_template
from main import *
from flask_cors import CORS
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pandas as pd
from rank_bm25 import BM25Okapi

import time


app = Flask(__name__)
CORS(app)
#Restaurant document data
print("Loading Started")
data = pd.read_pickle('restaurent_docs_ma.pickle')
documents = data['doc_information'].to_list()
document_ids = [i for i in range(len(documents))]
tokenized_docs = [doc.lower().split(" ") for doc in documents] 
bm25 = BM25Okapi(tokenized_docs)
print("Loading done")

# loader = DataFrameLoader(data, page_content_column="doc_information")
# bm25 = BM25Retriever.from_documents(loader.load())
# bm25.k = 10

# global api_key 
# api_key = os.environ["OPENAI_API_KEY"]
# embeddings = OpenAIEmbeddings(openai_api_key=api_key)
# vectordb = Chroma(persist_directory='./chroma_db_res2', embedding_function=embeddings)
# vector_retriever = vectordb.as_retriever(search_kwargs={"k": 5},search_type="mmr")

# vector_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# ensemble_retriever = EnsembleRetriever(
#     retrievers=[bm25, vector_retriever], weights=[0.5, 0.5]
# )



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.json["query"]
        zipcode = request.json["zipcode"]
        print("Processing Request")
        return jsonify(retrieval_info(data, document_ids, bm25, query, zipcode))
        # return jsonify(ensemble(llm,ensemble_retriever, query,zipcode))
    return jsonify({"message": "Welcome to the API"}), 200


if __name__ == '__main__':
    app.run(debug=True)