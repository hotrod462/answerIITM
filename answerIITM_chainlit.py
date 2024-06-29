#from pinecone import Pinecone, ServerlessSpec
import os
from pathlib import Path

from langchain_groq import ChatGroq

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import RunnablePassthrough


from dotenv import load_dotenv

import chainlit as cl

load_dotenv()

PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
GROQ_API_KEY= os.getenv("GROQ_API_KEY")
index_name= os.getenv("index_name")

@cl.on_chat_start
async def on_chat_start():
    # Sending an image with the local file path
    elements = [
    cl.Image(name="image1", display="inline", path=("./duck.png"))
    ]
    await cl.Message(content="Hello there, Welcome to answerIITM!", elements=elements).send()
    
    chat = ChatGroq(
    temperature=0,
    model="mixtral-8x7b-32768",
    api_key=GROQ_API_KEY, # Optional if not set as an environment variable,
    streaming=True
    )


    

    
    
    embedder= OpenAIEmbeddings(openai_api_base="http://localhost:1234/v1", openai_api_key="lm-studio", model="ChristianAzinn/e5-base-v2-gguf", embedding_ctx_length=1024, deployment="ChristianAzinn/e5-base-v2-gguf",check_embedding_ctx_length=False)

    
    
    ##Fix this when embeddings figured out
    
    # Load the Pinecone vector store
    vectorstore = await cl.make_async(PineconeVectorStore)(
        index_name=index_name, embedding=embedder, pinecone_api_key=PINECONE_API_KEY
    )
    print("Connected to pinecone")
    # Create a chain that uses the Pinecone vector store
    retriever= vectorstore.as_retriever()

    system_prompt = (
    "You are a helpful and friendly final year student at IIT Madras, one of the top tier colleges in india."
    "As you have been in the college for 4 years ,your job is to answer questions from students who have just cleared JEE Advanced and are now considering IIT Madras as one of the colleges to join"
    "Use simple english that everyone is able to understand, but do not oversimplify the technical terms in your answer"
    "If and when you do come up against technical terms, first make an effort to explain those terms using the context."
    "If the information is not present in the context, SAY CLEARLY THAT YOU DONT KNOW"
    "Give information in the form of a bulleted list when it is appropriate"
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Give as much detail as possible, as this information will be used for choosing the right college which will impact the students whole life"
    
    "\n\n"
    "{context}"
    "Check once more if youve provided enough detail about everything in the context given above, if not, add even more detail"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", system_prompt),
        ("human", "{question}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chat, prompt)
    rag_chain_1 = create_retrieval_chain(retriever, question_answer_chain)
    
    
    
    rag_chain_2 = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | chat
    
)

    cl.user_session.set("rag_chain_2", rag_chain_2)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("rag_chain_2")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res.content

    await cl.Message(content=answer).send()