import uvicorn
import aiofiles
import os
import logging
from urllib.parse import urlparse, parse_qs
from fastapi import FastAPI, UploadFile, HTTPException, status
from loguru import logger
from PathwayVectorStore.runVectorStore import run_vector_store
from main import pipeline
from src.config import (HOST_NAME, FAST_API_PORT)

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def extract_drive_id(drive_link: str) -> str:
    """Extract Google Drive ID from various URL formats while preserving original query param handling"""
    try:
        # Handle URL-encoded links
        if '?id=' in drive_link:
            return drive_link.split('?id=')[1].split('&')[0]
        
        # Handle standard folder/file URLs
        parts = [p for p in drive_link.split('/') if p]
        if 'd' in parts:
            return parts[parts.index('d') + 1]
        return parts[-1].split('?')[0]  # Remove query params
    except Exception as e:
        raise ValueError(f"Invalid Google Drive link format: {str(e)}")

@app.post("/upload")
async def save_file(file: UploadFile, drive_link: str) -> str:
    global collection_name_global
    collection_name_global = ""
    
    try:
        # Extract object ID from various Google Drive URL formats
        object_id = extract_drive_id(drive_link)
        if len(object_id) < 10:
            raise ValueError("Invalid Google Drive ID length")
            
        logger.success(f"Drive Link received: {drive_link}")
        logger.success(f"Validated Object ID: {object_id}")

        # Secure file write handling
        os.makedirs("./uploaded_files", exist_ok=True)
        temp_file_path = os.path.join("./uploaded_files", f"creds_{object_id}.json")
        
        # Async file write with cleanup
        async with aiofiles.open(temp_file_path, "wb") as f:
            await f.write(await file.read())
            
        logger.success(f"Credentials saved to: {temp_file_path}")

        # Start vector store processing
        run_vector_store(
            credential_path=temp_file_path,
            object_id=object_id
        )

        return ""

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.get("/process")
async def process_text(prompt: str, collection_name_global):
    # Original implementation preserved
    if collection_name_global is None:
        return {"error": "No collection name available. Please upload a file first."}
    logger.info(f"process coll name - {collection_name_global}")
    final_response = pipeline(
        query=prompt,
        topk=5,
        reranker=True,
        method="cr",
        agent_type="dynamic",
        use_reflection=True,
        n_reflection=1,
        use_router=True
    )
    return {"response_markdown": final_response}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST_NAME, port=FAST_API_PORT)
