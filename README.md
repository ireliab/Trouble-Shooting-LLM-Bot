# troubleshooting_llm

A retrieval-augmented (RAG) Q&A bot for answering troubleshooting questions from technical manuals, built to run against a self-hosted, OpenAI-compatible LLM endpoint (e.g. vLLM).

## Your documents
Put your manuals as Markdown files in `docs/`. They are not included in this repository.

## chunking_and_embedding.ipynb
1. Chunks the Markdown files by their headers (`MarkdownHeaderTextSplitter`), keeping the header path in each chunk
2. Embeds the chunks with `bge-small-en-v1.5` and stores them in a `FAISS` index in `vector_store/`

## main.py
QA workflow using an LLM and the vector store:
1. Loads the FAISS index
2. Builds a LangChain `RetrievalQA` chain
3. Retrieves the 10 most relevant chunks for the user's question
4. Passes them with the question to the LLM to generate the answer, served through a Gradio UI

## Configuration
Set these environment variables (e.g. in `.env`): `BASE_URL`, `API_KEY`, `MODEL_NAME`, `EMBED_MODEL_NAME`.

## Run
```bash
docker compose up --build
```
