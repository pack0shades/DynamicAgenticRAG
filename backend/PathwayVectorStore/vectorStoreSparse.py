import os
from loguru import logger
import pathway as pw
from typing import List, Dict
import numpy as np
# from pinecone_text.sparse import SpladeEncoder
from pymilvus.model import sparse
from tqdm import tqdm
import time
from pathway.xpacks.llm import parsers
from pathway.xpacks.llm.embedders import BaseEmbedder
from .vectorStore import VectorStoreServerModified
from .contextSplitter import ContextualRetrievalSplitter
from .config import (
    SPLADE_MODEL_NAME
)


# make sparse embedder
class SparseEmbedder(BaseEmbedder):
    def __init__(
        self,
        call_kwargs: dict = {},
    ):
        super().__init__()
        self.model = sparse.SpladeEmbeddingFunction(
            model_name= SPLADE_MODEL_NAME,
            device= "cpu"
        )   # model name is the model name of the splade model
        self.kwargs = call_kwargs

    def __wrapped__(self, input: str, **kwargs) -> np.ndarray:
        """
        Embed the text

        Args:
            - input: mandatory, the string to embed.
            - **kwargs: optional parameters for `encode` method. If unset defaults from the constructor
        will be taken. For possible arguments check
            `the Sentence-Transformers documentation
            <https://www.sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode>`_.
        """  # noqa: E501
        kwargs = {**self.kwargs, **kwargs}

        # make splade embedding
        splade_embedding = self.model.encode_documents([input])
        splade_embedding = splade_embedding.toarray()
        # make sparse representation to dense vector
        splade_embedding = splade_embedding.tolist()[0]

        return np.array(splade_embedding)



# make sparse vector store server
def make_sparse_vector_store_server(
    source, 
    port: int,
    save_doc_summary: bool, 
    save_doc_path: str = "", 
) -> None:

    parser = parsers.ParseUnstructured(
        mode='single'
    )

    embedder = SparseEmbedder()
    splitter = ContextualRetrievalSplitter()

    vector_server = VectorStoreServerModified(
        source,
        embedder=embedder,
        splitter=splitter,
        parser=parser,
        save_doc_summary=False,
        save_doc_path="",
        store_meta_data_in_chunk=True,
    )

    vector_server.run_server(
        host="127.0.0.1",
        port=port,
    )
    

# if __name__ == "__main__":

#     table = pw.io.gdrive.read(
#         object_id="",
#         service_user_credentials_file="./uploaded_files/credentials.json",
#         mode = "streaming",
#         with_metadata = True
#     )

#     # pw.run()

#     make_sparse_vector_store_server(
#         table,
#         port=8766,
#         save_doc_summary=False,
#         save_doc_path=""
#     )
