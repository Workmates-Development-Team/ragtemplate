# rag_utils.py
 
import os

import json

import boto3

import numpy as np

from dotenv import load_dotenv

from main import get_db_connection   # adjust this import if you move get_db_connection elsewhere
 
load_dotenv()
 
# Bedrock / embedding config (same as in your Flask app)

BEDROCK_REGION        = os.getenv("AWS_REGION", "us-east-1")

AWS_ACCESS_KEY        = os.getenv("AWS_ACCESS_KEY")

AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

MODEL_ID_EMBED        = os.getenv("MODEL_ID_EMBED")
 
# Initialize Bedrock client

_bedrock = boto3.client(

    "bedrock-runtime",

    region_name=BEDROCK_REGION,

    aws_access_key_id=AWS_ACCESS_KEY,

    aws_secret_access_key=AWS_SECRET_ACCESS_KEY

)
 
def embed_text_or_image(content, content_type="text", model_id=None):

    """

    Exactly the same embedding helper from your /upload route.

    """

    model_id = model_id or MODEL_ID_EMBED

    max_chars = 16000

    if content_type == "text" and len(content) > max_chars:

        content = content[:max_chars]
 
    if content_type == "text":

        body = json.dumps({

            "inputText": content,

            "dimensions": 256,

            "normalize": True

        })

    else:

        body = json.dumps({

            "inputImage": content,

            "dimensions": 256,

            "normalize": True

        })
 
    resp = _bedrock.invoke_model(

        modelId=model_id,

        body=body,

        accept="application/json",

        contentType="application/json"

    )

    data = json.loads(resp.get("body").read())

    return data.get("embedding")
 
 
def retrieve_similar_chunks(query_embedding, top_k=3):

    """

    Exactly the same retrieval logic from your /chat route.

    """

    conn, cursor = get_db_connection()

    cursor.execute(

        "SELECT chunk_index, embedding, file_name, file_path, chunk_text FROM embeddings"

    )

    rows = cursor.fetchall()

    cursor.close()

    conn.close()
 
    query_vec = np.array(query_embedding)

    sims = []

    for chunk_index, embedding, file_name, file_path, chunk_text in rows:

        if embedding is None:

            continue

        emb_vec = np.array(embedding)

        sim = float(np.dot(emb_vec, query_vec) / (np.linalg.norm(emb_vec) * np.linalg.norm(query_vec)))

        sims.append((sim, chunk_index, file_name, file_path, chunk_text))
 
    sims.sort(key=lambda x: x[0], reverse=True)

    return sims[:top_k]

 