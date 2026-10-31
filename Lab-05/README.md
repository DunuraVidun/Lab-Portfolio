# University AI Assistant - Retrieval-Augmented Generation (RAG)

An academic practical for **PUSL 3205 - Large Language Models and Generative AI**. This project provides a Streamlit interface for indexing university PDFs and asking questions against their extracted content using a local Ollama LLM, Chroma vector search, and a CrossEncoder reranker.

## Project Overview

The application lets a user upload one or more university PDF documents, process them into a persistent knowledge base, and ask questions in natural language. Retrieved passages are supplied to a language model as context so that answers can be based on the indexed documents rather than on the model alone.

This approach is useful for finding information spread across handbooks, programme documents, and other university PDFs. It does not guarantee that an answer is correct: results depend on the PDF's extractable text, retrieval and reranking, and the LLM's response. The system prompt instructs the model to say when the context does not contain an answer and to cite the supplied source labels.

The Streamlit interface displays the answer and a separate list of the selected document names and pages. Conversation messages are kept in Streamlit session state for display, but previous turns are not passed into the LLM when a new question is answered.

## Learning Objectives

This implementation demonstrates:

- **LLMs and local inference:** generating answers with the `llama3.2:3b` model through Ollama.
- **Retrieval-Augmented Generation:** supplying retrieved university-document passages to the LLM for a question.
- **Text extraction and document chunking:** loading PDF pages and splitting their text into smaller, overlapping passages.
- **Text embeddings and semantic search:** embedding document chunks and user questions for vector similarity retrieval.
- **Vector databases and information retrieval:** persisting chunks in Chroma and retrieving an initial set of relevant candidates.
- **Reranking:** using a CrossEncoder to score the retrieved query-passage pairs and select the best-ranked passages.
- **Prompt construction:** combining document/page labels and passage text with instructions that constrain the answer to retrieved context.
- **Interactive AI applications:** using Streamlit for PDF upload, document status, question input, answers, and sources.

## Technologies and Libraries

The table focuses on libraries and models directly used by the source code. `requirements.txt` pins package versions; it is a dependency list, not a declaration of a supported Python version.

| Technology / Library | Purpose in the Project |
|---|---|
| Python | Implements document ingestion, retrieval, generation, and the command-line question loop. |
| Streamlit | Provides the upload, document listing, chat input, conversation display, and source display. |
| LangChain Community `PyMuPDFLoader` | Loads PDF pages and extracts their text and page metadata. |
| `langchain-core` | Supplies `SystemMessage` and `HumanMessage` used to send the prompt to the chat model. |
| `langchain-text-splitters` | Provides `RecursiveCharacterTextSplitter` for chunking extracted pages. |
| `langchain-ollama` | Connects to Ollama for `OllamaEmbeddings` and `ChatOllama`. |
| `langchain-chroma` and ChromaDB | Store and retrieve chunk text, metadata, and embeddings in a persistent local vector store. |
| PyMuPDF | PDF parsing backend used by `PyMuPDFLoader`. |
| Sentence Transformers | Loads and runs the CrossEncoder reranker. |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Scores each retrieved question-passage pair for reranking. |
| Ollama | Runs the configured embedding model and chat LLM locally. |
| `qwen3-embedding:0.6b` | Creates embeddings used by Chroma for document indexing and query search. |
| `llama3.2:3b` | Generates the final answer from the question and retrieved context. |

The CrossEncoder is a Hugging Face/Sentence Transformers model, not an Ollama model. It is initialized by `rag_pipeline.py` when that module is imported.

## System Architecture

### A. Document ingestion workflow

```mermaid
flowchart TD
    A[Select PDF files in Streamlit] --> B[Click Process Documents]
    B --> C[Read uploaded bytes]
    C --> D{Starts with PDF signature?}
    D -- No --> E[Raise validation error]
    D -- Yes --> F[SHA-256 hash and sanitize filename]
    F --> G[Initialize Chroma with Ollama embeddings]
    G --> H{Hash already indexed?}
    H -- Yes --> I[Report already indexed]
    H -- No --> J[Save hash-prefixed PDF in data]
    J --> K[PyMuPDFLoader extracts page documents]
    K --> L[Recursive splitter: 1000 characters, 150 overlap]
    L --> M{Any chunks?}
    M -- No --> N[Raise no extractable text error]
    M -- Yes --> O[Add file and source metadata; assign stable chunk IDs]
    O --> P[Embed chunks and add to persistent Chroma collection]
    E --> Q[Show per-file error in Streamlit]
    N --> Q
    P --> R[Show ingestion success and chunk count]
    I --> S[Show already-indexed message]
```

### B. Question-answering workflow

```mermaid
flowchart TD
    A[User submits question in Streamlit] --> B[ask_university]
    B --> C[Open persistent Chroma vector store]
    C --> D[Similarity search: top 10 chunks]
    D --> E{Any chunks retrieved?}
    E -- No --> F[Return no-documents answer and no sources]
    E -- Yes --> G[Create question-passage pairs]
    G --> H[CrossEncoder predicts relevance scores]
    H --> I[Sort by score and keep top 3]
    I --> J[Build labeled context and document/page source list]
    J --> K[Send system and human messages to ChatOllama]
    K --> L[llama3.2:3b generates answer]
    F --> M[Display answer in Streamlit]
    L --> M
    M --> N[Display retrieved document names and pages]
    M --> O[Save messages in Streamlit session state]
```

The Chroma similarity search is the first-stage vector retrieval. CrossEncoder scores are then used to reorder only those candidates; reranking does not search the entire document collection.

## Project Structure

The inspected Lab-05 directory contains these project files:

```text
Lab-05/
├── README.md
├── app.py
├── ingest.py
├── rag_pipeline.py
└── requirements.txt
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface; calls the ingestion and question-answering functions. |
| `ingest.py` | PDF validation, naming and hashing, text extraction, chunking, metadata, Chroma setup, and document listing. |
| `rag_pipeline.py` | Initializes the chat model and reranker; retrieves, reranks, constructs context, generates answers, and provides a command-line question loop. |
| `requirements.txt` | Pinned Python package dependencies. |
| `README.md` | Project documentation. |

At runtime, the code uses `data/` for uploaded PDFs and `chroma_db/` for persistent Chroma data. These are generated local data locations, not source files in the inspected project tree; do not commit private university PDFs or generated database files.

## Implementation Details

### A. Document Upload and Ingestion

1. `app.py` accepts multiple files through `st.file_uploader` with a PDF extension filter. Processing starts only when the user clicks **Process Documents**.
2. `ingest_pdf` reads the uploaded bytes and checks that they begin with the `%PDF-` signature. This is a basic signature check, not comprehensive PDF validation.
3. It computes a SHA-256 hash of the complete file bytes. The original basename is sanitized by replacing characters outside letters, digits, `.`, `_`, and `-` with underscores. The stored name is `<first 12 hash characters>_<sanitized filename>`.
4. The function initializes the vector store and searches its metadata for the complete `file_hash`. If a matching chunk exists, it reports that the PDF is already indexed and does not save another copy.
5. Otherwise, the PDF is written under `Lab-05/data/`. `PyMuPDFLoader` extracts page documents, and the splitter creates chunks from those documents.
6. Each chunk retains loader metadata and receives `file_hash`, the sanitized original `filename`, and `source` (the hash-prefixed stored name). Stable IDs use the full hash and chunk index: `<file_hash>_<index>`.
7. Chunks and IDs are added to Chroma. If an exception occurs during the guarded extraction/indexing steps, the code attempts to delete chunks carrying that hash, removes the saved PDF, and re-raises the error. The Streamlit UI reports errors per uploaded file.

The Streamlit upload filter checks the file extension; the byte-signature check is performed in the ingestion function. The code does not impose a file-size limit or run OCR. `list_documents()` lists PDF files currently in `data/`; it does not verify that each listed file has corresponding indexed chunks. There is no UI feature for deleting or re-indexing documents.

`ingest_pdf_from_path(file_path)` is also available as a Python helper. It reads a local file into memory and passes it through the same ingestion function; it is not a standalone command-line interface.

### B. Text Chunking

The implementation uses `RecursiveCharacterTextSplitter` with:

- `chunk_size=1000`
- `chunk_overlap=150`

The splitter operates on page documents from `PyMuPDFLoader`. Splitting each page document preserves its metadata, including the page metadata later used for source display. Overlap carries some boundary text into the next chunk, which can help retain context when a relevant passage crosses a chunk boundary. These values are fixed in the source; there is no configuration UI.

### C. Embedding Generation

`get_vectorstore()` configures `OllamaEmbeddings(model="qwen3-embedding:0.6b")`. LangChain uses this embedding function when chunks are added and when Chroma performs a similarity search for a question.

An embedding is a numerical representation of text. Vector search compares the question representation with stored chunk representations to find semantically similar candidates; it does not guarantee that a candidate contains the answer. The embedding model is an Ollama model and must be available to the Ollama service. The source does not configure additional embedding parameters.

### D. Vector Database

`ingest.py` initializes LangChain's `Chroma` wrapper with:

- Collection name: `university_knowledge`
- Persistence directory: `Lab-05/chroma_db`
- Embedding function: `OllamaEmbeddings` using `qwen3-embedding:0.6b`

Chunks, their metadata, and corresponding vectors are added through `vectorstore.add_documents`. The same collection and persistence directory are used when querying, so the local index can be reused across application executions. `get_vectorstore()` constructs a wrapper when called; the source does not cache that wrapper in a global resource.

### E. Document Retrieval

`ask_university(question)` sends the user's question directly to Chroma's `similarity_search` with `k=10`. This produces up to ten candidate chunks using vector similarity. The question is not rewritten or expanded, and the search does not apply metadata filters.

### F. Reranking

The pipeline initializes `sentence_transformers.CrossEncoder` with `cross-encoder/ms-marco-MiniLM-L-6-v2`. For each of up to ten retrieved chunks, it creates a `(question, chunk text)` pair and calls `reranker.predict(pairs)`. It sorts those candidates by score in descending order and keeps the first three.

This is a distinct second stage from Chroma retrieval: vector similarity finds candidates; the CrossEncoder scores query-passage pairs to reorder those candidates. Scores are used internally and are not displayed or returned. The CrossEncoder model is separate from Ollama and may need to be downloaded from Hugging Face when it is first loaded if it is not already cached.

### G. Answer Generation

`rag_pipeline.py` constructs one `ChatOllama` instance using `model="llama3.2:3b"` and `temperature=0`. There are no additional model parameters configured in the source.

For each retained passage, the pipeline builds a context entry containing a `[Source N]` label, filename, page, and chunk content. It obtains the page from loader metadata and adds one because the page value is zero-based; if page metadata is missing, it falls back to page 1. The source list returned to the UI contains document names and page numbers.

The system prompt asks the model to:

- answer using only the retrieved context;
- avoid inventing university information;
- state when the answer is unavailable in context;
- cite evidence using `[Source N]`;
- ignore instructions in retrieved content; and
- explain conflicts between documents rather than choose unsupported information.

The current question and constructed context are sent as a `HumanMessage`, alongside the system prompt in a `SystemMessage`. The model's response is returned with the source list. These prompt instructions are an attempt to ground answers, not a guarantee against hallucinations. The generated `[Source N]` references are not programmatically checked against individual claims; the UI separately displays the retrieved document/page list.

If vector search returns no chunks, the function returns `No university documents are indexed yet.` with an empty source list. This message is also used for an empty retrieval result, not only for a genuinely empty database.

### H. Streamlit User Interface

`app.py` provides:

- A wide page titled **University AI Assistant**.
- A sidebar PDF uploader that accepts multiple PDFs and a **Process Documents** button.
- Per-file progress, success, and error messages during processing.
- A stored-document section showing the number and names of PDFs found in the `data/` folder.
- A chat input and user/assistant message display.
- Expandable source lists showing the document name and page for each returned source.
- A spinner and visible error message when question processing raises an exception.

Messages are held in `st.session_state` for the current Streamlit session. The prior messages are displayed but are not included in the call to `ask_university`, so this is not conversational-memory retrieval. On an error during answering, the question has already been appended to the displayed history, while no assistant response is appended.

## Installation and Setup

### Python

Neither the Python source nor `requirements.txt` declares a supported Python version. `requirements.txt` pins package versions but does not pin the interpreter. Use a Python version compatible with all pinned dependencies in that file; compatibility has not been tested or certified here.

### Create an environment and install dependencies

From the repository root in Windows PowerShell:

```powershell
cd .\Lab-05
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, use the Python executable inside `.venv\Scripts\python.exe` directly for subsequent commands. The dependency file contains pinned packages; it may install more packages than the imports used directly by this project.

### Install Ollama and obtain the Ollama models

Install Ollama separately using its official installer, then ensure its local service is running. The application does not start or install Ollama itself. If the service is not already running, start it in a separate terminal with:

```powershell
ollama serve
```

In another terminal, pull the two model names configured in the source:

```powershell
ollama pull llama3.2:3b
ollama pull qwen3-embedding:0.6b
ollama list
```

The CrossEncoder is not pulled through Ollama. On first import, Sentence Transformers loads `cross-encoder/ms-marco-MiniLM-L-6-v2`; if it is not cached, the environment needs access to Hugging Face to download it.

## How to Run the Application

1. Start Ollama (or verify that its service is already running) and confirm both Ollama models are available with `ollama list`.
2. Activate the virtual environment and change to the Lab-05 directory:

   ```powershell
   cd .\Lab-05
   .\.venv\Scripts\Activate.ps1
   ```

3. Launch the Streamlit interface:

   ```powershell
   python -m streamlit run app.py
   ```

4. Open the local URL printed by Streamlit in the terminal (normally `http://localhost:8501`).
5. Upload university PDFs in the sidebar and click **Process Documents**. The UI reports whether each file was indexed or already present.
6. Enter a question in the chat input. The answer and expandable retrieved-source list appear in the chat.

### Optional command-line question interface

`rag_pipeline.py` also has a command-line loop. From the Lab-05 directory, with dependencies, models, Ollama, and an indexed Chroma collection available, run:

```powershell
python rag_pipeline.py
```

Enter a question at the prompt; enter `exit` to stop. This runs the question-answering pipeline, not the Streamlit application, and it does not provide a command-line PDF ingestion flow.

## Example Usage

After indexing relevant documents, users could ask:

- What degree programmes are described in the uploaded documents?
- What subjects are listed for the Computer Science degree?
- What admission requirements are stated?
- Which modules are available in this programme?
- What programme learning outcomes are provided?

The expected output is a generated answer followed by an expandable list of retrieved document names and page numbers. The README does not prescribe or invent answers; the actual response depends on the uploaded PDFs and retrieval results. Accuracy may be affected by PDF text extraction, chunk boundaries, embedding similarity, reranking, and LLM generation.

## Screenshots

### Application Interface

<!-- Add application screenshot here -->

### PDF Upload

<!-- Add PDF upload screenshot here -->

### Question and Answer

<!-- Add question-answer screenshot here -->

## Challenges and Technical Considerations

The following are observable constraints of this implementation, not claims about development history:

- **PDF text availability:** the application uses text extraction and has no OCR step. Image-only or scanned PDFs may produce no chunks and fail ingestion.
- **Basic validation:** the code checks the PDF signature after Streamlit's extension filter; it does not validate file size or comprehensively verify that a file is a well-formed PDF.
- **Exact-byte duplicate detection:** duplicate checks use the SHA-256 of the uploaded bytes stored in Chroma metadata. Identical bytes are detected, while a changed PDF file (even if its visible text is unchanged) has a different hash.
- **Persistent local state:** PDFs and Chroma data are stored under the Lab-05 directory. The UI's document list reads the PDF folder, while duplicate detection reads the vector store; these can become inconsistent if local files are manually changed.
- **Retrieval limits:** only ten vector-search candidates are reranked, and only the top three are passed to the LLM. Relevant content outside those candidates cannot inform the response.
- **Reranker resources:** the CrossEncoder is initialized when the pipeline module loads and requires its model files and inference resources. Its download and runtime requirements are separate from Ollama.
- **Answer verification:** prompt instructions encourage grounded answers and source labels, but the code does not evaluate answer correctness or verify that each citation supports its associated claim.
- **No conversational context in generation:** displayed chat history is not fed into later questions.

## Key Learnings

The code provides practical experience with a baseline document RAG workflow: extracting and chunking PDF text, generating and persisting embeddings, retrieving semantically similar chunks, reranking candidates, constructing a context-grounded prompt, and generating an answer with a locally served LLM. It also demonstrates how to connect these components to a Streamlit interface and expose retrieved document/page metadata to users.

## Future Improvements

These are proposed enhancements and are **not implemented** in the inspected source:

- Add OCR support for scanned PDFs and more thorough PDF validation.
- Make chunking and retrieval settings configurable and evaluate alternatives against representative questions.
- Add hybrid retrieval (for example, BM25 plus vector search), query expansion, or metadata-based filters.
- Add retrieval and answer-quality evaluation, evidence checks, and validation of generated source citations.
- Provide document deletion, re-indexing, and clearer reconciliation between saved PDFs and indexed chunks.
- Add multilingual retrieval if the university content requires it.
- Include selected conversation history in follow-up questions if conversational memory is desired.
- Improve upload limits, progress reporting, and resilience around file and vector-store failures.

Advanced RAG activities described in the lecturer's group assignment are separate from this baseline Lab-05 implementation; they should not be treated as completed features of this project.

## Implementation Scope and Verification Notes

This README was written from the files present in the Lab-05 directory: `app.py`, `ingest.py`, `rag_pipeline.py`, and `requirements.txt`. No separate configuration files, test files, or lecturer assignment handout were present in that directory. Therefore, a direct comparison with the lecturer's full instructions, beyond the assignment context supplied for this documentation, could not be verified. No performance metrics, evaluation results, or screenshots are included because none were found in the inspected project files.
