# Notes and Reflection

## How the system works
This system uses a standard Retrieval-Augmented Generation (RAG) architecture built entirely in Python using `LangChain`. 
1. **Ingestion:** Documents are loaded using `PyPDFLoader` and `CSVLoader`, skipping overly complex files to respect time constraints. 
2. **Chunking & Indexing:** Text is split using `RecursiveCharacterTextSplitter` to maintain contextual chunks. These chunks are embedded using Nvidia's `nemotron-3-embed-1b` model and stored in a lightweight, local `ChromaDB` instance.
3. **Retrieval & Generation:** When a query comes in, the top 3 most relevant chunks are retrieved via vector similarity search. These chunks are injected into a strict system prompt and sent to Nvidia's `openai/gpt-oss-120b` endpoint to generate a factual answer.
4. **Guardrails:** A strict rule (`"If you don't know the answer, or the provided context does not contain the answer, just say exactly 'I CANNOT ANSWER'."`) is built into the prompt template to prevent hallucination.

## Next steps with more time
1. **Hybrid Search:** Currently, the system uses pure vector similarity. Budget queries often look for exact scheme names (e.g., "Nirbhaya Fund") or specific fiscal years. I would add BM25 (keyword search) to create a hybrid search retriever.
2. **Table Parsing / OCR:** The current PDFs are treated as flat text strings. I would integrate `unstructured` or `camelot` to properly extract tabular data so the AI can understand column hierarchies.
3. **Multi-language Support:** I would add an automatic translation layer (e.g., using a lightweight local model) to translate Hindi queries and Hindi budget text into English before embedding them, unifying the semantic space.

## AI Misstep Instance
During development, I initially tried to connect to the Nvidia API using the generic `OpenAIEmbeddings` class (since the Nvidia endpoint is often OpenAI-compatible). However, my AI assistant hallucinated that this generic wrapper would perfectly handle embeddings via the custom `base_url`. When executed, the Nvidia API threw a `400 Bad Request` because it expects a single string format, whereas the OpenAI wrapper sends a batch token sequence. I caught this by inspecting the error trace and immediately pivoted to installing the native `langchain-nvidia-ai-endpoints` package, which flawlessly resolved the structural mismatch.
