from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import requests
import os
import base64
from dotenv import load_dotenv
import uuid
import numpy as np
import faiss
import pickle
import json
from pathlib import Path

# Load environment variables
load_dotenv()

# Get API key from environment
EDEN_AI_API_KEY = os.getenv("EDEN_AI_API_KEY")
if not EDEN_AI_API_KEY:
    raise ValueError("EDEN_AI_API_KEY environment variable not set")

# Initialize FastAPI app
app = FastAPI(title="Image Embedding Search API", description="API for image embeddings and similarity search")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory vector database using FAISS
class FAISSVectorDB:
    def __init__(self, default_dimension=1024, index_file="image_faiss_index.pkl"):
        self.default_dimension = default_dimension
        self.index_file = index_file
        self.index = None
        self.metadata = []
        self.dimension = default_dimension
        self.is_initialized = False
        self.load_or_create_index()

    def load_or_create_index(self):
        if Path(self.index_file).exists():
            # Load existing index and metadata
            try:
                with open(self.index_file, "rb") as f:
                    saved_data = pickle.load(f)
                    self.index = saved_data["index"]
                    self.metadata = saved_data["metadata"]
                    self.dimension = saved_data.get("dimension", self.default_dimension)
                    self.is_initialized = True
                print(f"Loaded existing index with dimension {self.dimension} and {len(self.metadata)} items")
            except Exception as e:
                print(f"Error loading index: {str(e)}. Creating new index.")
                self.create_new_index(self.default_dimension)
        else:
            # Create a new index with default dimension
            self.create_new_index(self.default_dimension)
            
    def create_new_index(self, dimension):
        """Create a new FAISS index with specified dimension"""
        print(f"Creating new index with dimension {dimension}")
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        self.dimension = dimension
        self.is_initialized = True
            
    def save_index(self):
        """Save index and metadata to disk"""
        try:
            with open(self.index_file, "wb") as f:
                pickle.dump({
                    "index": self.index, 
                    "metadata": self.metadata,
                    "dimension": self.dimension
                }, f)
            print(f"Saved index with {self.index.ntotal} vectors and dimension {self.dimension}")
        except Exception as e:
            print(f"Error saving index: {str(e)}")

    def store_embedding(self, item_id: str, file_name: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """Store an embedding in the index"""
        # Check if we need to initialize the index with the correct dimension
        if not self.is_initialized or (self.index.ntotal == 0 and len(embedding) != self.dimension):
            print(f"Initializing index with dimension {len(embedding)} (was {self.dimension})")
            self.create_new_index(len(embedding))
        
        # Check dimension compatibility
        if len(embedding) != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: got {len(embedding)}, expected {self.dimension}")
            
        try:
            # Convert embedding to numpy array and reshape
            embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)
            
            # Add to index
            self.index.add(embedding_np)
            
            # Store metadata
            self.metadata.append({
                "id": item_id,
                "file_name": file_name,
                "metadata": metadata
            })
            
            # Save updated index and metadata
            self.save_index()
            
            return item_id
        except Exception as e:
            print(f"Error storing embedding: {str(e)}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        # Check if index is initialized
        if not self.is_initialized:
            return []
            
        # Check dimension compatibility
        if len(query_embedding) != self.dimension:
            raise ValueError(f"Query dimension mismatch: got {len(query_embedding)}, expected {self.dimension}")
        
        try:
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
                        "file_name": item["file_name"],
                        "metadata": item["metadata"],
                        "score": float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
                    })
            
            return results
        except Exception as e:
            print(f"Error during search: {str(e)}")
            raise

# Find embedding in a nested dictionary using recursive search
def find_embedding_in_dict(data, current_path=""):
    """Recursively search for an embedding field in a nested dictionary"""
    if isinstance(data, dict):
        # If this is a dictionary, check if it has an 'embedding' key
        if 'embedding' in data and isinstance(data['embedding'], list):
            return data['embedding'], current_path + ".embedding"
        
        # Otherwise, search in all values that are dictionaries or lists
        for key, value in data.items():
            path = current_path + "." + key if current_path else key
            result, path_found = find_embedding_in_dict(value, path)
            if result:
                return result, path_found
    
    elif isinstance(data, list):
        # If this is a list, search in all items that are dictionaries or lists
        for i, item in enumerate(data):
            path = f"{current_path}[{i}]"
            result, path_found = find_embedding_in_dict(item, path)
            if result:
                return result, path_found
    
    # No embedding found in this branch
    return None, ""

# Initialize vector database
db = FAISSVectorDB()

# File upload endpoints
@app.post("/embeddings/upload")
async def create_embedding_from_upload(
    file: UploadFile = File(...),
    provider: str = Form("google"),
    model: Optional[str] = Form(None),
    representation: str = Form("document")
):
    """
    Generate embedding for an uploaded image file and store it in the vector database
    """
    try:
        url = "https://api.edenai.run/v2/image/embeddings"
        
        # Prepare provider parameter (with model if specified)
        provider_param = f"{provider}/{model}" if model else provider
        
        # Read file content
        file_content = await file.read()
        
        # Create multipart form data
        files = {
            "file": (file.filename, file_content, file.content_type)
        }
        
        data = {
            "providers": provider_param,
            "response_as_dict": "true",
            "attributes_as_list": "false",
            "show_original_response": "true",
            "representation": representation
        }
        
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {EDEN_AI_API_KEY}"
        }
        
        # Make API call
        print(f"Calling Eden AI for uploaded image: {file.filename}")
        response = requests.post(url, files=files, data=data, headers=headers)
        
        # Log the full response for debugging
        print("Full API Response:")
        response_data = response.json()
        
        # Find the provider key in the response
        provider_key = None
        for key in response_data.keys():
            if key.startswith(provider):
                provider_key = key
                break
            
        if not provider_key:
            # Dump all available keys
            print(f"Available top-level keys: {list(response_data.keys())}")
            raise ValueError(f"Provider key for '{provider}' not found in response")
            
        # Get the provider-specific response
        provider_response = response_data[provider_key]
        print(f"Provider response keys: {list(provider_response.keys())}")
        
        # Use recursive search to find embedding anywhere in the response
        embedding = None
        embedding_path = ""
        
        # First try in the provider_response
        embedding, embedding_path = find_embedding_in_dict(provider_response)
        
        # If not found, try in the original_response
        if not embedding and "original_response" in provider_response:
            embedding, embedding_path = find_embedding_in_dict(provider_response["original_response"])
            if embedding:
                embedding_path = "original_response" + embedding_path
                
        # If still not found and we have items, check each item directly
        if not embedding and "items" in provider_response and isinstance(provider_response["items"], list):
            for i, item in enumerate(provider_response["items"]):
                if "embedding" in item and isinstance(item["embedding"], list):
                    embedding = item["embedding"]
                    embedding_path = f"items[{i}].embedding"
                    break
        
        if not embedding:
            # Save response to file for debugging
            with open("eden_ai_response.json", "w") as f:
                json.dump(response_data, f, indent=2)
            print("Full response saved to eden_ai_response.json")
            
            # Raise error
            raise ValueError("No embedding found in response. Full response saved to eden_ai_response.json")
        
        print(f"Found embedding at path: {embedding_path}")
        print(f"Embedding length: {len(embedding)}")
        
        # Generate unique ID
        item_id = str(uuid.uuid4())
        
        # Store in vector database
        db.store_embedding(
            item_id=item_id,
            file_name=file.filename,
            embedding=embedding,
            metadata={"filename": file.filename}
        )
        
        # Return response
        return {
            "id": item_id,
            "file_name": file.filename,
            "metadata": {"filename": file.filename},
            "embedding_length": len(embedding),
            "embedding_path": embedding_path
        }
            
    except Exception as e:
        print(f"Error processing uploaded image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/search/upload")
async def search_by_uploaded_image(
    file: UploadFile = File(...),
    provider: str = Form("google"),
    model: Optional[str] = Form(None),
    representation: str = Form("query"),
    top_k: int = Form(5)
):
    """
    Search for similar images using an uploaded image
    """
    try:
        url = "https://api.edenai.run/v2/image/embeddings"
        
        # Prepare provider parameter (with model if specified)
        provider_param = f"{provider}/{model}" if model else provider
        
        # Read file content
        file_content = await file.read()
        
        # Create multipart form data
        files = {
            "file": (file.filename, file_content, file.content_type)
        }
        
        data = {
            "providers": provider_param,
            "response_as_dict": "true",
            "attributes_as_list": "false",
            "show_original_response": "true",
            "representation": representation
        }
        
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {EDEN_AI_API_KEY}"
        }
        
        # Make API call
        response = requests.post(url, files=files, data=data, headers=headers)
        
        # Log the full response for debugging
        print("Full API Response for search:")
        response_data = response.json()
        
        # Find the provider key
        provider_key = None
        for key in response_data.keys():
            if key.startswith(provider):
                provider_key = key
                break
        
        if not provider_key:
            print(f"Available top-level keys: {list(response_data.keys())}")
            raise ValueError(f"Provider key for '{provider}' not found in response")
        
        provider_response = response_data[provider_key]
        
        # Use recursive search to find embedding anywhere in the response
        embedding = None
        embedding_path = ""
        
        # First try in the provider_response
        embedding, embedding_path = find_embedding_in_dict(provider_response)
        
        # If not found, try in the original_response
        if not embedding and "original_response" in provider_response:
            embedding, embedding_path = find_embedding_in_dict(provider_response["original_response"])
            if embedding:
                embedding_path = "original_response" + embedding_path
                
        # If still not found and we have items, check each item directly
        if not embedding and "items" in provider_response and isinstance(provider_response["items"], list):
            for i, item in enumerate(provider_response["items"]):
                if "embedding" in item and isinstance(item["embedding"], list):
                    embedding = item["embedding"]
                    embedding_path = f"items[{i}].embedding"
                    break
        
        if not embedding:
            # Save response to file for debugging
            with open("eden_ai_search_response.json", "w") as f:
                json.dump(response_data, f, indent=2)
            print("Full search response saved to eden_ai_search_response.json")
            
            # Raise error
            raise ValueError("No embedding found in search response")
        
        print(f"Found query embedding at path: {embedding_path}")
        print(f"Query embedding length: {len(embedding)}")
        
        # Search vector database
        results = db.search(embedding, top_k)
        
        return {
            "query_image": file.filename,
            "results": results
        }
        
    except Exception as e:
        print(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")
    
@app.post("/embeddings/upload-multiple")
async def create_embeddings_from_multiple_uploads(
    files: List[UploadFile] = File(...),
    provider: str = Form("google"),
    model: Optional[str] = Form(None),
    representation: str = Form("document")
):
    """
    Generate embeddings for multiple uploaded image files and store them in the vector database
    """
    results = []
    errors = []
    
    # Process each file
    for file in files:
        try:
            url = "https://api.edenai.run/v2/image/embeddings"
            
            # Prepare provider parameter (with model if specified)
            provider_param = f"{provider}/{model}" if model else provider
            
            # Read file content
            file_content = await file.read()
            
            # Create multipart form data
            files_data = {
                "file": (file.filename, file_content, file.content_type)
            }
            
            data = {
                "providers": provider_param,
                "response_as_dict": "true",
                "attributes_as_list": "false",
                "show_original_response": "true",
                "representation": representation
            }
            
            headers = {
                "accept": "application/json",
                "authorization": f"Bearer {EDEN_AI_API_KEY}"
            }
            
            # Make API call
            print(f"Calling Eden AI for uploaded image: {file.filename}")
            response = requests.post(url, files=files_data, data=data, headers=headers)
            response_data = response.json()
            
            # Find the provider key in the response
            provider_key = None
            for key in response_data.keys():
                if key.startswith(provider):
                    provider_key = key
                    break
                
            if not provider_key:
                raise ValueError(f"Provider key for '{provider}' not found in response")
                
            # Get the provider-specific response
            provider_response = response_data[provider_key]
            
            # Use recursive search to find embedding
            embedding = None
            embedding_path = ""
            
            # First try in the provider_response
            embedding, embedding_path = find_embedding_in_dict(provider_response)
            
            # If not found, try in the original_response
            if not embedding and "original_response" in provider_response:
                embedding, embedding_path = find_embedding_in_dict(provider_response["original_response"])
                if embedding:
                    embedding_path = "original_response" + embedding_path
                    
            # If still not found and we have items, check each item directly
            if not embedding and "items" in provider_response and isinstance(provider_response["items"], list):
                for i, item in enumerate(provider_response["items"]):
                    if "embedding" in item and isinstance(item["embedding"], list):
                        embedding = item["embedding"]
                        embedding_path = f"items[{i}].embedding"
                        break
            
            if not embedding:
                raise ValueError(f"No embedding found in response for {file.filename}")
            
            print(f"Found embedding at path: {embedding_path}")
            print(f"Embedding length: {len(embedding)}")
            
            # Generate unique ID
            item_id = str(uuid.uuid4())
            
            # Store in vector database
            db.store_embedding(
                item_id=item_id,
                file_name=file.filename,
                embedding=embedding,
                metadata={"filename": file.filename}
            )
            
            # Add to results
            results.append({
                "id": item_id,
                "file_name": file.filename,
                "embedding_length": len(embedding),
                "status": "success"
            })
            
            # Reset file position for potential reuse
            await file.seek(0)
            
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            errors.append({
                "file_name": file.filename,
                "error": str(e),
                "status": "error"
            })
            # Continue with next file
            
    return {
        "successful": results,
        "failed": errors,
        "total_processed": len(results),
        "total_failed": len(errors)
    }

# Get info about the vector database
@app.get("/info")
async def get_database_info():
    """Get information about the vector database"""
    try:
        return {
            "status": "active",
            "dimension": db.dimension,
            "num_vectors": db.index.ntotal if db.index else 0,
            "num_items": len(db.metadata),
            "initialized": db.is_initialized
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)