
import gradio as gr
import base64
from datetime import datetime
import os
import sys
import time
import uuid
from PIL import Image as PIL_Image
import fitz
# Initialize Vertex AI libraries for working with generative s
from google.cloud import aiplatform
# Import LangChain components
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
import regex as re
import shutil
from google.cloud import storage
import json
import ast


# Initialize Vertex AI
import vertexai
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    Image,
    Part,
    Tool,
)
from vertexai.language_models import TextEmbeddingModel
import vertexai.preview.generative_models as generative_models
from vertexai.preview.generative_models import ToolConfig
from dotenv import load_dotenv
import os
import logging
import subprocess


load_dotenv()
logging.basicConfig(level=logging.ERROR)


def save_runtime_env(index_id, curr_index_endpoint, index_name, bucket_location, rag_unique_identifier):
    """Persist runtime Vertex AI vector-search metadata to the project .env file."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    runtime_values = {
        "INDEX_ID": index_id,
        "INDEX_ENDPOINT": curr_index_endpoint,
        "CURRENT_INDEX_ENDPOINT": curr_index_endpoint,
        "INDEX_NAME": index_name,
        "BUCKET_LOCATION": bucket_location,
        "RAG_UNIQUE_IDENTIFIER": rag_unique_identifier,
        "curr_index_endpoint": curr_index_endpoint,
        "index_id": index_id,
        "index_name": index_name,
        "bucket_location": bucket_location,
        "rag_unique_identifier": rag_unique_identifier,
    }

    existing = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                existing[key.strip()] = value.strip().strip('"')

    for key, value in runtime_values.items():
        if value is not None:
            existing[key] = str(value).strip()

    with open(env_path, "w", encoding="utf-8") as f:
        for key in sorted(existing):
            value = existing[key]
            f.write(f'{key}="{value}"\n')

    os.environ["INDEX_ID"] = str(index_id)
    os.environ["INDEX_ENDPOINT"] = str(curr_index_endpoint)
    os.environ["INDEX_NAME"] = str(index_name)
    os.environ["BUCKET_LOCATION"] = str(bucket_location)
    os.environ["RAG_UNIQUE_IDENTIFIER"] = str(rag_unique_identifier)


PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")


print(f"PROJECT_ID: {PROJECT_ID}")
print(f"LOCATION: {LOCATION}")

# Defines the Generative Models Configuration
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0,
    "top_p": 0.95,
}

# Loading Gemini Model
multimodal_model = GenerativeModel(
    "gemini-2.5-flash", generation_config=generation_config)

# Initializing embedding model
text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")



def create_clean_image_folder(Image_Path):
    # Create the directory if it doesn't exist
    if not os.path.exists(Image_Path):
        os.makedirs(Image_Path)
    image_star = Image_Path + "*"
    if os.path.isdir(image_star):
        shutil.rmtree(image_star)
    elif os.path.isfile(image_star):
        os.remove(image_star)
    
def split_pdf_extract_data(pdfList, folder_uri):
    # To get better resolution
    zoom_x = 2.0  # horizontal zoom
    zoom_y = 2.0  # vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

    for indiv_Pdf in pdfList:
        doc = fitz.open(indiv_Pdf)  # open document
        for page in doc:  # iterate through the pages
            pix = page.get_pixmap(matrix=mat)  # render page to an image
            outpath = f"{folder_uri}{indiv_Pdf}_{page.number}.png"
            pix.save(outpath)  # store image as a PNG

    # Define the path where images are located
    image_names = os.listdir(folder_uri)
    Max_images = len(image_names)

    # Create empty lists to store image information
    page_source = []
    page_content = []
    page_id = []

    p_id = 0  # Initialize image ID counter
    rest_count = 0  # Initialize counter for error handling

    while p_id < Max_images:
        try:
            # Construct the full path to the current image
            image_path = folder_uri + image_names[p_id]

            # Load the image
            image = Image.load_from_file(image_path)

            # Generate prompts for text and table extraction
            prompt_text = "Extract all text content in the image"
            prompt_table = (
                "Detect table in this image. Extract content maintaining the structure"
            )
            prompt_image = "Detect images in this image. Extract content in the form of alternative text or subtitles to each sub-image"

            # Extract text using your multimodal model
            contents = [image, prompt_text]
            response = multimodal_model.generate_content(contents)
            text_content = response.text

            # Extract table using your multimodal model
            contents = [image, prompt_table]
            response = multimodal_model.generate_content(contents)
            table_content = response.text

            # Extract information from images (i.e. Subtitle / Alternative text). | Currently Disabled
            # contents = [image, prompt_image]
            # response = multimodal_model.generate_content(contents)
            # image_content = response.text

            # Log progress and store results
            print(f"processed image no: {p_id}")
            page_source.append(image_path)
            page_content.append(
                text_content + "\n" + table_content
            )  # + "\n" + image_content)
            page_id.append(p_id)
            p_id += 1

        except Exception as err:
            # Handle errors during processing
            print(err)
            print("Taking Some Rest")
            time.sleep(
                12
            )  # Pause execution for 12 second due to default Quota for Vertex AI
            rest_count += 1
            if rest_count == 5:  # Limit consecutive error handling
                rest_count = 0
                print(f"Cannot process image no: {image_path}")
                p_id += 1  # Move to the next image

    # Create a DataFrame to store extracted information
    df = pd.DataFrame(
        {"page_id": page_id, "page_source": page_source, "page_content": page_content}
    )
    del page_id, page_source, page_content  # Conserve memory
    df.head()  # Preview the DataFrame

    return df


def generate_text_embedding(text) -> list:
    """Text embedding with a Large Language Model."""
    embeddings = text_embedding_model.get_embeddings([text])
    vector = embeddings[0].values
    return vector

# Returns a chunked embeddings dataframe


def create_chunked_embeddings(df):
    # Create a DataFrameLoader to prepare data for LangChain
    loader = DataFrameLoader(df, page_content_column="page_content")

    # Load documents from the 'page_content' column of your DataFrame
    documents = loader.load()

    # Log the number of documents loaded
    print(f"# of documents loaded (pre-chunking) = {len(documents)}")

    # Create a text splitter to divide documents into smaller chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=10000,  # Target size of approximately 10000 characters per chunk
        chunk_overlap=200,  # overlap between chunks
    )

    # Split the loaded documents
    doc_splits = text_splitter.split_documents(documents)

    # Add a 'chunk' ID to each document split's metadata for tracking
    for idx, split in enumerate(doc_splits):
        split.metadata["chunk"] = idx

    # Log the number of documents after splitting
    print(f"# of documents = {len(doc_splits)}")

    texts = [doc.page_content for doc in doc_splits]
    text_embeddings_list = []
    id_list = []
    page_source_list = []
    for doc in doc_splits:
        id = uuid.uuid4()
        text_embeddings_list.append(generate_text_embedding(doc.page_content))
        id_list.append(str(id))
        page_source_list.append(doc.metadata["page_source"])
        time.sleep(12)  # So that we don't run into Quota Issue

    # Creating a dataframe of ID, embeddings, page_source and text
    embedding_df = pd.DataFrame(
        {
            "id": id_list,
            "embedding": text_embeddings_list,
            "page_source": page_source_list,
            "text": texts,
        }
    )
    embedding_df.head()
    return embedding_df

def create_json_file(embedding_df, RAG_unique_identifier):
    # save id and embedding as a json file
    json_file_name = RAG_unique_identifier + ".json"
    jsonl_string = embedding_df[["id", "embedding"]].to_json(
        orient="records", lines=True
    )
    with open(json_file_name, "w") as f:
        f.write(jsonl_string)

    # Show the first few lines of the json file
    #! head -n 3 {json_file_name}
    return json_file_name


def upload_file_to_gcs(json_file_name, bucket_location):
    # Generates a unique ID for session
    #UID = datetime.now().strftime("%m%d%H%M%S")
    # Creates a GCS bucket
    #BUCKET_URI = f"gs://{bucket_location}--{UID}"

    BUCKET_URI = "vec-search-bucket-aquastride--0808174253"

    
    client = storage.Client(project=PROJECT_ID)

    bucket = client.bucket(f"{BUCKET_URI}", user_project=PROJECT_ID)
   
    #bucket = client.create_bucket(bucket)

    blob = bucket.blob(json_file_name)
    blob.upload_from_filename(json_file_name)
    
    return f"gs://{BUCKET_URI}"

def create_index(vec_search_index_name, bucket_location):
    return aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=f"{vec_search_index_name}",
        contents_delta_uri=bucket_location,
        dimensions=768,
        approximate_neighbors_count=20,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=7,

    )

def create_Index_Endpoint(my_index, vec_search_index_name):
    my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=f"{vec_search_index_name}",
        public_endpoint_enabled=True,
    )

    DEPLOYED_INDEX_NAME = vec_search_index_name.replace(
        "-", "_"
    )  # Can't have '-' in deployment name, only alphanumeric and _ allowed
    UID = datetime.now().strftime("%m%d%H%M%S")
    DEPLOYED_INDEX_ID = f"{DEPLOYED_INDEX_NAME}_{UID}"
    # deploy the Index to the Index Endpoint
    my_index_endpoint.deploy_index(index=my_index, deployed_index_id=DEPLOYED_INDEX_ID)

    return my_index_endpoint, DEPLOYED_INDEX_ID

def Test_LLM_Response(txt):
    """
    Determines whether a given text response generated by an LLM indicates a lack of information.

    Args:
        txt (str): The text response generated by the LLM.

    Returns:
        bool: True if the LLM's response suggests it was able to generate a meaningful answer,
              False if the response indicates it could not find relevant information.

    This function works by presenting a formatted classification prompt to the LLM (`gemini_pro_model`).
    The prompt includes the original text and specific categories indicating whether sufficient information was available.
    The function analyzes the LLM's classification output to make the determination.
    """

    classification_prompt = f""" Classify the text as one of the following categories:
        -Information Present
        -Information Not Present
        Text=The provided context does not contain information.
        Category:Information Not Present
        Text=I cannot answer this question from the provided context.
        Category:Information Not Present
        Text:{txt}
        Category:"""
    classification_response = multimodal_model.generate_content(
        classification_prompt
    ).text

    if "Not Present" in classification_response:
        return False  # Indicates that the LLM couldn't provide an answer
    else:
        return True  # Suggests the LLM generated a meaningful response


def get_prompt_text(question, context):
    """
    Generates a formatted prompt string suitable for a language model, combining the provided question and context.

    Args:
        question (str): The user's original question.
        context (str): The relevant text to be used as context for the answer.

    Returns:
        str: A formatted prompt string with placeholders for the question and context, designed to guide the language model's answer generation.
    """
    prompt = """
      Answer the question using the context below. Respond with only information from the text provided
      Question: {question}
      Context : {context}
      """.format(
        question=question, context=context
    )
    return prompt


def get_answer(
    embedding_df, my_index_endpoint, DEPLOYED_INDEX_ID, query="No Query was provided."
):
    """
    Retrieves an answer to a provided query using multimodal RAG.

    This function leverages a vector search system to find relevant text documents from a
    pre-indexed store of multimodal data. Then, it uses a large language model (LLM) to generate
    an answer, using the retrieved documents as context.

    Args:
        query (str): The user's original query.

    Returns:
        dict: A dictionary containing the following keys:
            * 'result' (str): The LLM-generated answer.
            * 'neighbor_index' (int): The index of the most relevant document used for generation
                                     (for fetching image path).

    Raises:
        RuntimeError: If no valid answer could be generated within the specified search attempts.
    """

    neighbor_index = 0  # Initialize index for tracking the most relevant document
    answer_found_flag = 0  # Flag to signal if an acceptable answer is found
    result = ""  # Initialize the answer string
    # Use a default image if the reference is not found
    page_source = "./no-matching-pages.png"  # Initialize the blank image
    query_embeddings = generate_text_embedding(
        query
    )  # Generate embeddings for the query

    response = my_index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_embeddings],
        num_neighbors=5,
    )  # Retrieve up to 5 relevant documents from the vector store

    while answer_found_flag == 0 and neighbor_index < 4:
        context = embedding_df[
            embedding_df["id"] == response[0][neighbor_index].id
        ].text.values[
            0
        ]  # Extract text context from the relevant document

        prompt = get_prompt_text(
            query, context
        )  # Create a prompt using the question and context
        result = multimodal_model.generate_content(
            prompt
        ).text  # Generate an answer with the LLM

        if Test_LLM_Response(result):
            answer_found_flag = 1  # Exit loop when getting a valid response
        else:
            neighbor_index += (
                1  # Try the next retrieved document if the answer is unsatisfactory
            )

    if answer_found_flag == 1:
        page_source = embedding_df[
            embedding_df["id"] == response[0][neighbor_index].id
        ].page_source.values[
            0
        ]  # Extract image_path from the relevant document
    return result, page_source


def build_embedding(RAG_unique_identifier, rag_list_pdfs):
    # Creates a Unique folder for the segmented PDF images. (Each page of the PDF is converted into a .PNG)
        folder_url = f"./{RAG_unique_identifier}_images/"
        create_clean_image_folder(folder_url)
    
        # Creates the embeddings dataframe of the PDF Images.
        company_dataframe = split_pdf_extract_data(rag_list_pdfs, folder_url)
        company_embeddings_dataframe = create_chunked_embeddings(company_dataframe)
    
        # Creates unique names for the Google Cloud Vector Search & GCS Bucket URL.
        vec_search_index_name = f"vec-search-index-{RAG_unique_identifier}"
        bucket_name = f"vec-search-bucket-{RAG_unique_identifier}"
    
        # Uploads the embeddings to GCS as a JSON file.
        json_file_name = create_json_file(
            company_embeddings_dataframe, RAG_unique_identifier
        )
        bucket_location = upload_file_to_gcs(json_file_name, bucket_name)
    
        print(f"Bucket Location: {bucket_location} and file uploaded to json" )

        return vec_search_index_name, bucket_location, company_embeddings_dataframe



def create_RAG(RAG_unique_identifier, rag_list_pdfs):
    
    build_embedding(RAG_unique_identifier, rag_list_pdfs)
    folder_url = f"./{RAG_unique_identifier}_images/"
    create_clean_image_folder(folder_url)

    ###check if the embeddings CSV file already exists, if so, load it instead of creating it again
    if os.path.exists(f"{RAG_unique_identifier}_embeddings.csv"):
        company_embeddings_dataframe = pd.read_csv(
            f"{RAG_unique_identifier}_embeddings.csv"
        )
        company_embeddings_dataframe["embedding"] = company_embeddings_dataframe["embedding"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    else:
        # Creates the embeddings dataframe of the PDF Images.
        company_dataframe = split_pdf_extract_data(rag_list_pdfs, folder_url)
        company_embeddings_dataframe = create_chunked_embeddings(company_dataframe)

        ### save the embeddings dataframe to a CSV file for future reference
        if len(company_embeddings_dataframe) != 0:
            company_embeddings_dataframe.to_csv(
                f"{RAG_unique_identifier}_embeddings.csv", index=False
            )

    # Creates unique names for the Google Cloud Vector Search & GCS Bucket URL.
    vec_search_index_name = f"vec-search-index-{RAG_unique_identifier}"
    bucket_name = f"vec-search-bucket-{RAG_unique_identifier}"

    # Uploads the embeddings to GCS as a JSON file.
    json_file_name = create_json_file(
        company_embeddings_dataframe, RAG_unique_identifier
    )
    def validate_embedding_jsonl(df):
        for row in df[["id", "embedding"]].to_dict("records"):
            if not isinstance(row["embedding"], list):
                raise TypeError(f"Invalid embedding payload: {row}")
    validate_embedding_jsonl(company_embeddings_dataframe)
           
    bucket_location = upload_file_to_gcs(json_file_name, bucket_name)
    #vec-search-bucket-aquastride--0808174253
    #vec-search-bucket-aquastride--0808174253

    print(f"Bucket Location: {bucket_location} and file uploaded to json" )

    # This function may take up to 25 minutes to run to deploy the custom Vector Search to a Public Endpoint.
    index = create_index(vec_search_index_name, bucket_location)
    my_index_endpoint, index_id = create_Index_Endpoint(index, vec_search_index_name)

    curr_index_endpoint = getattr(my_index_endpoint, "resource_name", None)
    if curr_index_endpoint is None:
        curr_index_endpoint = getattr(my_index_endpoint, "name", None)

    save_runtime_env(
        index_id=index_id,
        curr_index_endpoint=curr_index_endpoint,
        index_name=vec_search_index_name,
        bucket_location=bucket_location,
        rag_unique_identifier=RAG_unique_identifier,
    )

    # Create a reusable Object for each Rag Model to call upon
    RAG_model_info = {
        "bucket_uri": bucket_location,
        "index": index,
        "embeddings_dataframe": company_embeddings_dataframe,
        "index_id": index_id,
        "my_index_endpoint": my_index_endpoint,
    }

    return RAG_model_info


# Needs to be lowercase characters with no spaces; e.g. "test", "aquastride".
RAG_unique_identifier = "aquastride"  # @param {type: "string"}

# List the PDFs to be processed via the RAG Endpoint.
pdf_list = ["aquastride_company.pdf", "aquastride_DB.pdf"]

# Creates the RAG model endpoint on Vertex AI Vector Search.
rag_info = create_RAG(RAG_unique_identifier, pdf_list)


#Provide a Query to test the deployed endpoint.
# Highly recommended to use a call to a Database (i.e. Cloud SQL) with the extracted Serial number.
query = "Provided the Serial_No (CZE5F6G7) and SKU (DepthStrider_23_Red_Norm), Determine the cx_name who purchased this serial number.\n Output the Owner (cx_name) and the address (cx_address) in this format: \nThank you [cx_name] for your purchase! We have you on file at [cx_address]."  # @param {type: "string"}

# Responds with the result of the query against the RAG endpoint & its source.
result, page_source = get_answer(
    rag_info["embeddings_dataframe"],
    rag_info["my_index_endpoint"],
    rag_info["index_id"],
    query,
)

# If the endpoint returns irrelevant context to the LLM, respond with the below.
if page_source == "./no-matching-pages.png":
    result = (
        "I could not find your answer within the Data. Can you rephrase your question?"
    )

# Print the results and it's page source.
print(f"Response: {result}\nPage Source: {page_source}")

#########################################################
