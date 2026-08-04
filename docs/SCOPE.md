# Scope

In this 2-hour assignment, the primary goal was to build a functional, end-to-end Retrieval-Augmented Generation (RAG) pipeline that prioritizes reliability and handles failure gracefully.

## What was included
We successfully ingested and processed 10 out of the 11 provided files, including 9 PDFs (e.g., `gender_budget_2024-25.pdf`) and the `MRF_13_Union_Budget.csv`. These were parsed using `PyPDFLoader` and `CSVLoader`, chunked using `RecursiveCharacterTextSplitter`, and embedded into a local `ChromaDB` using Nvidia's `nemotron-3-embed-1b` model.

## What was excluded and why
I deliberately configured the ingestion pipeline to skip files that crashed the loader, prioritizing pipeline completion over perfect ingestion.
Specifically, `stat20.xls` was skipped. The unstructured loader required additional dependencies (like `networkx` and `openpyxl`) that were missing. Given the strict time constraint, I chose not to go down the rabbit hole of debugging Excel parsers and instead allowed the `try/except` block to catch the error, skip the file, and continue processing the rest of the corpus.

## What was handled imperfectly
1. **Tables in PDFs**: The PDFs were read as raw text. Any complex tables inside the government budgets were likely flattened into a single stream of text, losing their column-row relationships. 
2. **CSV Context**: The `CSVLoader` treats each row as a separate document. While it extracts the text, it loses the broader context of what the spreadsheet as a whole represents. 
3. **Bilingual Text**: Documents with both English and Hindi text were embedded as-is. While modern embedding models can handle multilingual text decently, translating the Hindi portions to English before embedding would likely yield much higher retrieval accuracy.
