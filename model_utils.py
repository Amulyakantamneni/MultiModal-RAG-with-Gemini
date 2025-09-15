from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from pathlib import Path
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.schema.messages import HumanMessage
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from google import genai
from pdf_processor import embed_text

load_dotenv()

def gemini_image_prompt(image_path_or_file, prompt_text, model_name="gemini-2.5-flash", api_key=None):
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    print(image_path_or_file)
    image = Image.open(image_path_or_file)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, prompt_text]
        )

    return response.text

# from end_to_end_projetcs.cataract_multimodal_rag.main import vector_db
def create_vectordb(docs_with_embeddings):
    vector_db = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb)
                         for doc, emb in docs_with_embeddings],
        embedding=None,
        metadatas=[doc.metadata for doc, _ in docs_with_embeddings]
    )
    return vector_db


def save_vectordb(vector_db, save_path="outputs/vectorstore"):
    Path(save_path).mkdir(parents=True, exist_ok=True)

    vector_db.save_local(save_path)
    print(f"vector_db saved to {save_path}")


def retrieve_doc(query, vector_db, k=5):
    query_embedding = embed_text(query)
    top_docs = vector_db.similarity_search_by_vector(
        embedding=query_embedding,
        k=k
    )
    return top_docs


def load_vector_db(load_path="outputs/vectorstore"):
    print(f"🔍 Checking path: {load_path}")
    print(f"📁 Directory exists: {Path(load_path).exists()}")

    if not Path(load_path).exists():
        print(f"❌ Vector store not found at {load_path}")
        return None

    # Check what files are in the directory
    if Path(load_path).exists():
        files = list(Path(load_path).iterdir())
        print(f"📄 Files in directory: {files}")

    try:
        vector_db = FAISS.load_local(
            load_path,
            embeddings=None,
            allow_dangerous_deserialization=True
        )
        print("✅ Vector db loaded successfully")
        return vector_db
    except Exception as e:
        print(f"❌ Could not load vectordb \n Reason: {e}")
        return None


def create_multimodal_query(query, retrieved_docs, image_data_store):
    content = []

    content.append({
        "type": "text",
        "text": f"The User has asked the query {query} \n\nContext:\n"
    })

    image_docs = []
    text_docs = []

    for docs in retrieved_docs:
        # print(docs)
        if docs.metadata.get("type") == "text":
            text_docs.append(docs)
        else:
            image_docs.append(docs)

    for text_doc in text_docs:
        page_content = text_doc.page_content
        # print(page_content)
        content.append({
            "type": "text",
            "text": f"Excerpts from the retrieved docs \n {page_content}"
        })

    for image_doc in image_docs:
        page_num = image_doc.metadata.get("page")
        # print(page_num)
        image_id = image_doc.metadata.get("image_id")

        content.append({
            "type": "text",
            "text": f"Image is taken from page number {page_num}"
        })

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_data_store[image_id]}"
            }
        })

        content.append({
            "type": "text",
            "text": " Give the response based on the provide text and images "
        })

    return HumanMessage(content=content)


def initialize_llm():
    load_dotenv()
    llm = GoogleGenerativeAI(model="gemini-2.5-flash",
                             api_key=os.getenv("GOOGLE_API_KEY"))
    return llm


def get_llm_output(llm, llm_query):
    response = llm.invoke([llm_query])
    return response
