import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain_and_retriever():
    # 1. Load the database we created in ingest.py
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"Database folder '{CHROMA_PATH}' not found. Did you run ingest.py first?")
        
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 2. Setup the Retriever (the search engine)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 3. Setup the AI Model (using Nvidia Endpoint)
    llm = ChatNVIDIA(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0,
        timeout=180
    )

    # 4. Setup the Prompt with GUARDRAILS
    system_prompt = (
        "You are an assistant for answering questions about Indian government budgets. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, or the provided context does not contain the answer, "
        "just say exactly 'I CANNOT ANSWER'. Do not try to make up an answer.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. Connect it all together using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def ask_question(question, chain_tuple):
    rag_chain, retriever = chain_tuple
    print(f"\nQuestion: {question}")
    
    # Get the answer
    answer = rag_chain.invoke(question)
    
    # Get the sources manually since we used LCEL
    docs = retriever.invoke(question)
    
    sources = []
    if answer != "I CANNOT ANSWER":
        for doc in docs:
            source_file = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", -1)
            source_file = os.path.basename(source_file)
            sources.append({"file": source_file, "page": page})
            
    try:
        print(f"Answer: {answer}")
    except UnicodeEncodeError:
        print(f"Answer: {answer.encode('ascii', 'replace').decode('ascii')}")
    if sources:
        print(f"Sources: {sources}")
        
    return answer, sources

if __name__ == "__main__":
    print("Loading AI Model and Database...")
    try:
        chain_tuple = get_rag_chain_and_retriever()
        while True:
            user_input = input("\nAsk a question (or type 'quit' to exit): ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            ask_question(user_input, chain_tuple)
            
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
