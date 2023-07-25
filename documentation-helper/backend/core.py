import os
from typing import Any, Dict, List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone

from consts import INDEX_NAME

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

def run_llm(query:str, chat_history: List[Dict[str, Any]]=[])->any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    #qa = RetrievalQA.from_chain_type(llm=chat,chain_type="stuff", retriever=docsearch.as_retriever(),return_source_documents =True)
    qa = ConversationalRetrievalChain.from_llm(llm=chat,retriever=docsearch.as_retriever(), return_source_documents = True)
    return qa({"question":query, "chat_history":chat_history})

if __name__ == "__main__":
    print(run_llm("What is langchain?"))
