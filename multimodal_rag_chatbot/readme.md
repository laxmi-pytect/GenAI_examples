# Multimodal RAG Warranty Chatbot

This project contains a two-step implementation for an AquaStride warranty-assistant chatbot:

- `part1.py` builds the retrieval layer by ingesting PDF content, extracting page text, creating embeddings, uploading them to GCS, and creating a Vertex AI Matching Engine index and endpoint.
- `part2.py` loads the existing runtime metadata from `.env`, retrieves relevant document chunks from the deployed index, and drives the actual multimodal chatbot flow with Gradio and Gemini.

The system combines:

- PDF ingestion and page rendering
- multimodal image analysis with Gemini
- text embeddings from Vertex AI
- vector search with Matching Engine
- retrieval-augmented generation (RAG)
- a user-facing warranty workflow in Gradio

## High-level architecture

The app follows a split workflow:

1. Build the vector knowledge base from AquaStride documents.
2. Save the resulting Vertex AI metadata to `.env`.
3. Load the saved metadata in the chat application.
4. Convert user questions to embeddings.
5. Search the nearest document chunks in the vector index.
6. Pass those chunks as context to Gemini for grounded answers.
7. Support image-based warranty flow using multimodal function calling.

## Part 1: Knowledge-base creation

File: `part1.py`

This script creates the RAG index and saves runtime config values used later by the chatbot.

### What it does

- loads environment values via `load_dotenv()`
- defines `save_runtime_env(...)` to persist runtime values into `.env`
- converts PDF pages into PNG images via PyMuPDF
- calls Gemini multimodal models to extract page text and table content
- creates a DataFrame of page content and source files
- splits documents into chunks using `CharacterTextSplitter`
- generates embeddings with `TextEmbeddingModel.from_pretrained("text-embedding-005")`
- exports the embeddings to JSONL and uploads them to Google Cloud Storage
- creates a Vertex AI Matching Engine index and endpoint
- saves values like:
  - `INDEX_ID`
  - `INDEX_ENDPOINT`
  - `INDEX_NAME`
  - `BUCKET_LOCATION`
  - `RAG_UNIQUE_IDENTIFIER`

### Key functions in part1

- `save_runtime_env(index_id, curr_index_endpoint, index_name, bucket_location, rag_unique_identifier)`
  - writes the index metadata to `.env`
- `split_pdf_extract_data(pdfList, folder_uri)`
  - extracts text from each image-rendered PDF page
- `generate_text_embedding(text)`
  - returns an embedding vector for a query or chunk
- `create_chunked_embeddings(df)`
  - creates chunk-level records and embeddings
- `create_json_file(embedding_df, RAG_unique_identifier)`
  - writes the JSONL payload used by Matching Engine
- `upload_file_to_gcs(json_file_name, bucket_location)`
  - uploads the embeddings to GCS
- `create_index(vec_search_index_name, bucket_location)`
  - creates the vector index
- `create_Index_Endpoint(my_index, vec_search_index_name)`
  - creates and deploys the index endpoint

## Part 2: Chatbot runtime

File: `part2.py`

This script loads the saved runtime values from `.env` and runs the chatbot.

### What it does

- loads `.env` using `load_dotenv()`
- reads values such as:
  - `INDEX_ID`
  - `INDEX_ENDPOINT`
  - `INDEX_NAME`
  - `BUCKET_LOCATION`
  - `RAG_UNIQUE_IDENTIFIER`
- creates a `MatchingEngineIndexEndpoint` from the saved endpoint resource name
- loads the saved embedding CSV if present and converts stringified embeddings back to actual arrays
- embeds the user query with the same embedding model
- runs `find_neighbors(...)` against the vector index
- extracts the matching chunk from the local embedding dataframe
- sends the matched document text as context to Gemini for an answer
- validates whether the answer is meaningful
- supports multimodal warranty flows using function calls and Gradio

### Key functions in part2

- `generate_text_embedding(text)`
  - generates a query embedding
- `Test_LLM_Response(txt)`
  - checks whether the LLM response contains meaningful answer content
- `get_prompt_text(question, context)`
  - builds the grounded answer prompt
- `get_answer(embedding_df, my_index_endpoint, DEPLOYED_INDEX_ID, query)`
  - retrieves nearest neighbors and asks Gemini to respond from the context
- `convert_image_for_analysis(image)`
  - converts uploaded images into Gemini-compatible image parts
- `flow_manager(current_function_call)`
  - routes image-based and text-based warranty tasks to the right workflow case
- `bot(message, history)`
  - main Gradio chatbot logic

## Multimodal workflow

The user journey is:

1. User starts the chat.
2. They may upload an image of the shoe tag.
3. The image is processed by Gemini to extract serial number and SKU.
4. The app generates a retrieval prompt from the extracted values.
5. The app queries the vector DB to find matching company/customer context.
6. Gemini responds with a grounded answer.
7. The user can then upload a damage image for warranty assessment.
8. The app determines whether the damage is covered or not covered.
9. The flow may continue into return-shipping or support recommendations.

## Runtime data flow

```text
part1.py
  -> PDF extraction
  -> chunking + embeddings
  -> upload embeddings to GCS
  -> create Vertex AI index + endpoint
  -> write metadata to .env

part2.py
  -> read metadata from .env
  -> load endpoint and embeddings
  -> query nearest neighbors
  -> send context to Gemini
  -> answer and route to workflow
  -> Gradio chat interface
```

## Environment variables

The project uses a `.env` file with values such as:

```env
GOOGLE_GENAI_USE_VERTEXAI="1"
GOOGLE_CLOUD_PROJECT="your-project-id"
GOOGLE_CLOUD_LOCATION="us-central1"
INDEX_ID="<runtimevalue>"
INDEX_ENDPOINT="<runtimevaluegenerated>"
INDEX_NAME="<runtimevalue>/userprovided"
BUCKET_LOCATION="<runtimevalue>"
RAG_UNIQUE_IDENTIFIER="<runtimevalue>/userprovided"
```

## File structure

```text
multimodal_chatbot/
├── .env
├── part1.py
├── part2.py
├── aquastride_company.pdf
├── aquastride_DB.pdf
├── aquastride_embeddings.csv
├── aquastride.json
├── aquastride_images/
├── my_shoe_tag.png
├── damaged_shoe.png
├── uploaded_image.png
├── chatbot.ipynb
├── readme.md
└── __pycache__/
```

## Summary

This project is a two-part multimodal RAG chatbot:

- `part1.py` builds and stores the vector knowledge base.
- `part2.py` loads that saved state and runs the actual customer support experience.

Together, they create a warranty-support assistant that can answer policy questions, verify order information from product tags, and assess damage using multimodal prompts grounded in retrieved context from AquaStride documents.
