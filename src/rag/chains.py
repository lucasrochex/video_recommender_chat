from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from pinecone import Pinecone, ServerlessSpec

import os


load_dotenv()

# Models
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Embedding model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") # LLM model

# Cloud Vector Store along with embedding model
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "school-of-life-index"  # Pre populated index
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


def interpret_user_question(question):
    # User question analyzer for embedding
    prompt_analyzer = ChatPromptTemplate.from_messages(
        [("system", 
        "Write a summary, main ideas and keywords of the human inquiry below. Pin point day to day problems the inquiry is suggesting"),
        ("human", "{question}")]
    )
    # Instantiate chain
    analyzer_chain = LLMChain(llm=llm, prompt=prompt_analyzer)
    result = analyzer_chain.invoke({"question": question})
    enhanced_question = result['text']
    return enhanced_question


def perform_similarity_search(query, n_docs=5):
    results = vector_store.similarity_search(
        query,
        k=n_docs,
        #filter={"source": "tweet"},sharp_mcclintock
    )
    result_string = ""
    for res in results:
        result_string += f"* {res.page_content} [{res.metadata}]"

    return result_string

def generate_answer(original_question, relevant_content):
    prompt_answer = ChatPromptTemplate.from_messages(
        [("system", 
        "You are a philosofy video recomendation tool. Below you will receive the summary of video main ideas and the query that led to these recomendations. Explain very brefiely why each the video is recommended and provide the link by combining https://www.youtube.com/watch?v= and the youtube_id field. Use all videos. Structure answer in markdown. Answer in portuguese please."),
        ("user", "User initial question {question} \n Retrieval system recs: {recs}")]
    )

    chain_answer = LLMChain(llm=llm, prompt=prompt_answer)
    final_answer = chain_answer.invoke({"question":original_question, "recs": relevant_content})
    return final_answer


def get_video_recommendation(question):
    enhanced_question = interpret_user_question(question)    
    relevant_content = perform_similarity_search(enhanced_question)
    final_answer = generate_answer(question, relevant_content)
    return final_answer['text']
    