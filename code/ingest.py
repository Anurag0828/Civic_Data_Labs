import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Load environment variables
load_dotenv()

# We need an NVIDIA API key to use their endpoints
if not os.getenv("NVIDIA_API_KEY"):
    print("WARNING: NVIDIA_API_KEY not found in environment variables.")
    print("Please create a .env file and add your key, or set it manually.")
    exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, "..", "rag_assignment", "corpus")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def get_loader_for_file(file_path):
    ext = file_path.lower().split('.')[-1]
    if ext == 'pdf':
        return PyPDFLoader(file_path)
    elif ext == 'csv':
        return CSVLoader(file_path)
    elif ext in ['xls', 'xlsx']:
        try:
            # Need pandas/openpyxl for excel, sometimes UnstructuredExcelLoader is finicky
            from langchain_community.document_loaders import UnstructuredExcelLoader
            return UnstructuredExcelLoader(file_path)
        except ImportError:
            print("To process Excel files, ensure you have unstructured and pandas installed.")
            return None
    else:
        print(f"Unsupported file format: {ext}")
        return None

def main():
    print("Starting ingestion...")
    all_docs = []
    
    # Grab all files in the corpus directory
    files = glob.glob(os.path.join(CORPUS_PATH, "*.*"))
    print(f"Found {len(files)} files in {CORPUS_PATH}/")
    
    for file_path in files:
        print(f"Processing: {file_path}")
        loader = get_loader_for_file(file_path)
        
        if not loader:
            print(f"Skipping {file_path} due to missing loader.")
            continue
            
        try:
            # Load the document
            docs = loader.load()
            all_docs.extend(docs)
            print(f"  -> Successfully loaded {len(docs)} pages/rows.")
        except Exception as e:
            # THIS IS CRITICAL FOR THE ASSIGNMENT:
            # We skip messy files and document them later in SCOPE.md
            print(f"  -> FAILED to load {file_path}. Error: {str(e)[:100]}...")
            print(f"  -> Skipping this file for now to save time.")
            
    print(f"\nTotal loaded documents/pages: {len(all_docs)}")
    if not all_docs:
        print("No documents were loaded. Exiting.")
        return

    # Split documents into smaller chunks
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks.")
    
    # Save chunks to Vector DB
    print("\nCreating ChromaDB embeddings (this might take a minute)...")
    embeddings_model = NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    db = Chroma.from_documents(
        chunks,
        embeddings_model,
        persist_directory=CHROMA_PATH
    )
    print(f"Successfully saved embeddings to {CHROMA_PATH}/")

if __name__ == "__main__":
    main()
