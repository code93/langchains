from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
import pinecone

import os

# os.environ.get("OPENAI_API_KEY")

pinecone.init(api_key="1e905a28-1d51-41f4-ab0d-7d56746389e3",environment="us-west4-gcp-free")

if __name__ == '__main__':
    print("hello vectorstore")
    loader = TextLoader("/home/lightbearer/github/intro-to-vector-db/mediumblogs/mediumblog1.txt")
    document = loader.load()
    # print(document)
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(texts,embeddings,index_name="mediun-blogs-embeddings-index")

   # qa = VectorDBQA.from_chain_type(llm=OpenAI(),chain_type="stuff",vectorstore=docsearch, return_source_documents=True)
    qa = VectorDBQA.from_chain_type(llm=OpenAI(),chain_type="stuff",vectorstore=docsearch)

    query = "What is a vector DB? Give me a 15 word answer for a begginner."
    result = qa({"query":query})
    print(result)