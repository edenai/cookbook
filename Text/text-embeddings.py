from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import requests
import os
from dotenv import load_dotenv
import uuid
import numpy as np
import faiss
import pickle
from pathlib import Path

# Load environment variables
load_dotenv()

# Get API key from environment
EDEN_AI_API_KEY = os.getenv("EDEN_AI_API_KEY")
if not EDEN_AI_API_KEY:
    raise ValueError("EDEN_AI_API_KEY environment variable not set")

# Initialize FastAPI app
app = FastAPI(title="Embedding Search API", description="API for text embeddings and similarity search")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models
class TextItem(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TextItems(BaseModel):
    items: List[TextItem]
    provider: str = "openai"  # Default provider
    model: Optional[str] = None  # Optional specific model

class SearchQuery(BaseModel):
    query: str
    provider: str = "openai"
    model: Optional[str] = None
    top_k: int = 5

class EmbeddingResponse(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: List[float]

# Simple in-memory vector database using FAISS
class FAISSVectorDB:
    def __init__(self, dimension=1536, index_file="faiss_index.pkl"):
        self.dimension = dimension
        self.index_file = index_file
        self.index = None
        self.metadata = []
        self.load_or_create_index()

    def load_or_create_index(self):
        if Path(self.index_file).exists():
            # Load existing index and metadata
            with open(self.index_file, "rb") as f:
                saved_data = pickle.load(f)
                self.index = saved_data["index"]
                self.metadata = saved_data["metadata"]
        else:
            # Create a new index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            
    def save_index(self):
        # Save index and metadata to disk
        with open(self.index_file, "wb") as f:
            pickle.dump({"index": self.index, "metadata": self.metadata}, f)

    def store_embedding(self, item_id: str, text: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        # Convert embedding to numpy array and reshape
        embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)
        
        # Add to index
        self.index.add(embedding_np)
        
        # Store metadata
        self.metadata.append({
            "id": item_id,
            "text": text,
            "metadata": metadata
        })
        
        # Save updated index and metadata
        self.save_index()
        
        return item_id
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        # Convert query to numpy array
        query_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # If index is empty, return empty results
        if self.index.ntotal == 0:
            return []
            
        # Ensure top_k doesn't exceed the number of vectors in the index
        top_k = min(top_k, self.index.ntotal)
        
        # Search the index
        distances, indices = self.index.search(query_np, top_k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):  # Safety check
                item = self.metadata[idx]
                results.append({
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "score": float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
                })
        
        return results

# Initialize vector database
db = FAISSVectorDB()

async def get_embedding_for_single_text(text: str, provider: str, model: Optional[str] = None) -> List[float]:
    """Get embedding for a single text to avoid batching issues"""
    url = "https://api.edenai.run/v2/text/embeddings"
    
    # Prepare provider parameter (with model if specified)
    provider_param = f"{provider}/{model}" if model else provider
    
    payload = {
        "providers": [provider_param],
        "texts": [text],  # Just one text at a time
        "response_as_dict": True,
        "attributes_as_list": False,
        "show_original_response": True
    }
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {EDEN_AI_API_KEY}"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        
        # Find the provider key
        provider_key = None
        for key in response_data.keys():
            if key.startswith(provider):
                provider_key = key
                break
        
        if not provider_key or "items" not in response_data[provider_key]:
            raise ValueError(f"No valid response for {provider}")
            
        # Extract the embedding
        items = response_data[provider_key]["items"]
        if not items or "embedding" not in items[0]:
            raise ValueError("No embedding found in response")
            
        return items[0]["embedding"]
        
    except Exception as e:
        print(f"Error getting embedding for text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting embedding: {str(e)}")

@app.post("/embeddings", response_model=List[EmbeddingResponse])
async def create_embeddings(items: TextItems):
    """
    Generate embeddings for a list of texts and store them in the vector database
    """
    # Extract texts
    texts = [item.text for item in items.items]
    result = []
    
    # Process each text individually to avoid batching issues
    for i, item in enumerate(items.items):
        try:
            print(f"Processing text {i+1}/{len(items.items)}: {item.text[:50]}...")
            
            # Get embedding for this text
            embedding = await get_embedding_for_single_text(
                item.text, 
                items.provider, 
                items.model
            )
            
            # Generate unique ID
            item_id = str(uuid.uuid4())
            
            # Store in vector database
            db.store_embedding(
                item_id=item_id,
                text=item.text,
                embedding=embedding,
                metadata=item.metadata
            )
            
            # Add to result
            result.append(EmbeddingResponse(
                id=item_id,
                text=item.text,
                metadata=item.metadata,
                embedding=embedding
            ))
            
            print(f"Successfully processed text {i+1}")
            
        except Exception as e:
            print(f"Error processing text {i+1}: {str(e)}")
            # Continue with other texts
    
    return result

@app.post("/search")
async def search(query: SearchQuery):
    """
    Search for similar texts using semantic similarity
    """
    try:
        # Get embedding for search query (using the new single-text method)
        query_embedding = await get_embedding_for_single_text(query.query, query.provider, query.model)
        
        # Search vector database
        results = db.search(query_embedding, query.top_k)
        
        return {
            "query": query.query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
