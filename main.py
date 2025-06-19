from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import ollama
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
import os
from typing import Optional

load_dotenv()

collection_name = "IntentDetection6"

class ChatRequest(BaseModel):
    prompt: str
    similarity_threshold: float = 0.5  

# Weaviate Client Setup 
def initialize_weaviate() -> Optional[weaviate.WeaviateClient]:
    """Initialize and return Weaviate client"""
    try:
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_key = os.getenv("WEAVIATE_API_KEY")
        
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_key),
        )
        print(f"Successfully connected to Weaviate: {client.is_ready()}")
        print(f"Using collection: {collection_name}")
        
        return client
            
    except Exception as e:
        print(f"Error connecting to Weaviate: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.weaviate_client = initialize_weaviate()
    yield
    # Shutdown: Close Weaviate client
    if hasattr(app.state, 'weaviate_client') and app.state.weaviate_client is not None:
        app.state.weaviate_client.close()
        print("Weaviate client closed.")

# Create a FastAPI App
app = FastAPI(
    title="Intent Detection",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Dependency to get Weaviate client
def get_weaviate_client() -> Optional[weaviate.WeaviateClient]:
    """Dependency to get Weaviate client from app state"""
    return getattr(app.state, 'weaviate_client', None)

def search_similar_prompts(
    query_prompt: str, 
    threshold: float = 0.5, 
    limit: int = 1,
    client: Optional[weaviate.WeaviateClient] = None) -> Optional[dict]:
    """
    Search for similar prompts in Weaviate using cosine similarity.
    Returns the most similar prompt-response pair if similarity > threshold.
    """
    if client is None:
        return None
    
    try:
        collection = client.collections.get(collection_name)
        
        # Perform vector search
        response = collection.query.near_text(
            query=query_prompt,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
            return_properties=["prompt", "response"],
            distance=0.8
        )
        
        if response.objects:
            most_similar = response.objects[0]
            similarity = 1 - most_similar.metadata.distance
            
            print(f"Found similar prompt with similarity: {similarity:.3f}")
            print(f"Original prompt: '{most_similar.properties['prompt']}'")
            
            if similarity >= threshold:
                return {
                    "prompt": most_similar.properties["prompt"],
                    "response": most_similar.properties["response"],
                    "similarity": similarity,
                    "uuid": str(most_similar.uuid)
                }
        return None
        
    except Exception as e:
        print(f"Error searching for similar prompts: {e}")
        return None

# API Endpoint
@app.post("/chat")
def chat_with_ollama(
    request: ChatRequest,
    weaviate_client: Optional[weaviate.WeaviateClient] = Depends(get_weaviate_client)
    ):
    """
    This endpoint receives a prompt, checks for similar prompts in Weaviate,
    and either returns cached response or gets a new response from Ollama.
    """
    user_prompt = request.prompt
    similarity_threshold = request.similarity_threshold

    print(f"Received prompt: '{user_prompt}'")
    print(f"Similarity threshold: {similarity_threshold}")
    
    # Checking for similar prompts
    if weaviate_client is not None:
        similar_result = search_similar_prompts(
            user_prompt, 
            similarity_threshold, 
            client=weaviate_client
        )
        
        if similar_result:
            print(f"Found similar prompt! Returning cached response (similarity: {similar_result['similarity']:.3f})")
            
            # Store the new prompt with cached response
            weaviate_id = None
            try:
                collection = weaviate_client.collections.get(collection_name)
                result = collection.data.insert({
                    "prompt": user_prompt,
                    "response": similar_result["response"],
                })
                weaviate_id = str(result)
                print(f"Stored cached conversation in Weaviate with ID: {weaviate_id}")
            except Exception as e:
                print(f"Error storing cached response in Weaviate: {e}")
            
            return {
                "prompt": user_prompt,
                "response": similar_result["response"],
                "status": "cached_response",
                "similar_prompt": similar_result["prompt"],
                "similarity": similar_result["similarity"],
                "cached_from_uuid": similar_result["uuid"],
                "stored_in_weaviate": weaviate_id is not None,
                "weaviate_id": weaviate_id
            }

    # Get Response from Ollama 
    print("No similar prompt found. Calling Ollama...")
    try:
        # Send prompt to Ollama model
        response_data = ollama.chat(
            model='intent',
            messages=[{'role': 'user', 'content': user_prompt}],
        )
        ollama_response = response_data['message']['content']
        print("Received response from Ollama.")
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with Ollama: {e}")

    # Store the prompt-response pair in Weaviate
    weaviate_id = None
    if weaviate_client is not None:
        try:
            collection = weaviate_client.collections.get(collection_name)
            result = collection.data.insert({
                "prompt": user_prompt,
                "response": ollama_response,
            })
                
            weaviate_id = str(result)
            print(f"Stored conversation in Weaviate with ID: {weaviate_id}")
                
        except Exception as e:
            print(f"Error storing in Weaviate: {e}")

    # Return the response
    response_data = {
        "prompt": user_prompt,
        "response": ollama_response,
        "status": "new_response"
    }
    
    # Include Weaviate ID if successfully stored
    if weaviate_id:
        response_data["weaviate_id"] = weaviate_id
        response_data["stored_in_weaviate"] = True
    else:
        response_data["stored_in_weaviate"] = False
    
    return response_data

# Check if API is running
@app.get("/")
def read_root():
    return {
        "message": "FastAPI is running",
        "endpoints": {
            "chat": "POST /chat - Main chat endpoint with similarity search",
        }
    }