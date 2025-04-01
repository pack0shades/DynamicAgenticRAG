import pathway as pw
import threading
from loguru import logger
import time
from typing import Optional
from .vectorRetriever import VectorStoreRetriever
from .vectorStoreDense import make_dense_vector_store_server
from .vectorStoreSparse import make_sparse_vector_store_server

def wait_for_server_ready(port: int, max_retries: int = 30, backoff_factor: float = 1.5) -> bool:
    """Wait until server is ready to accept connections with exponential backoff."""
    retries = 0
    while retries < max_retries:
        try:
            client = VectorStoreRetriever("localhost", port)
            num_files = client.get_num_input_files()
            if num_files > 0:
                logger.info(f"Server on port {port} ready with {num_files} files")
                return True
        except Exception as e:
            logger.debug(f"Attempt {retries+1} failed on port {port}: {str(e)}")
            sleep_time = backoff_factor ** retries
            time.sleep(sleep_time)
            retries += 1
    logger.error(f"Server on port {port} not ready after {max_retries} retries")
    return False

def run_vector_store(
    credential_path: str,
    object_id: str,
    dense_port: int = 8765,
    sparse_port: int = 8766,
    summary_path: str = "./document_data/document_summary.txt"
) -> None:
    """Main function to initialize and run vector store servers with health checks."""
    
# Read table with context manager for resource cleanup
    table = pw.io.gdrive.read(
        object_id=object_id,
        service_user_credentials_file=credential_path,
        mode = "streaming",
        with_metadata = True
    )
    
    # Configure and start servers
    servers = [
        threading.Thread(
            target=make_dense_vector_store_server,
            args=(table, dense_port, True, summary_path),
            daemon=True
        ),
        threading.Thread(
            target=make_sparse_vector_store_server,
            args=(table, sparse_port, False, ""),
            daemon=True
        )
    ]

    for server in servers:
        server.start()

    logger.info("Both servers initiated")

    # Wait for servers to become ready
    if not all([
        wait_for_server_ready(dense_port),
        wait_for_server_ready(sparse_port)
    ]):
        raise RuntimeError("Failed to start one or more servers")

    logger.info("Both servers are ready to accept requests")

    # Keep main thread alive while servers are running
    return ""

if __name__ == "__main__":
    run_vector_store(
        credential_path="./uploaded_files/credentials2.json",
        object_id="1nbZbC7ccg2JCcAuI2JX4-FARo9kbnPKU"
    )
