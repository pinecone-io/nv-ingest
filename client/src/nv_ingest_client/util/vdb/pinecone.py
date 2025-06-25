from nv_ingest_client.util.vdb.adt_vdb import VDB
from typing import Any, Dict, List, Optional
from pinecone import Pinecone, ServerlessSpec

class PineconeVDB(VDB):
    """
    Pinecone implementation of the VDB interface using Pinecone Python SDK v6.0.0+.
    Supports both dense and sparse vector indexes.
    """
    def __init__(
        self,
        api_key: str,
        cloud: str = "aws",
        region: str = "us-east-1",
        index_name: str = "nv-ingest-index",
        dimension: int = 1536,
        metric: str = "cosine",
        sparse: bool = False,
        **kwargs,
    ):
        """
        Initialize Pinecone client and set up index parameters.
        Args:
            api_key (str): Pinecone API key.
            cloud (str): Cloud provider (e.g., 'aws', 'gcp').
            region (str): Cloud region (e.g., 'us-east-1').
            index_name (str): Name of the Pinecone index.
            dimension (int): Vector dimension.
            metric (str): Similarity metric ('cosine', 'euclidean', etc.).
            sparse (bool): Whether to use sparse vectors (for hybrid search).
            **kwargs: Additional keyword arguments.
        """
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.sparse = sparse
        self.kwargs = kwargs
        # Create index if it doesn't exist
        if not self.pc.list_indexes().names or index_name not in self.pc.list_indexes().names:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
                **({"sparse": {"enabled": True}} if sparse else {}),
            )
        self.index = self.pc.Index(index_name)

    # ----------------------
    # Index Management
    # ----------------------
    def create_index(self, **kwargs):
        """
        Create a Pinecone index if it does not exist.
        """
        if not self.pc.list_indexes().names or self.index_name not in self.pc.list_indexes().names:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                **({"sparse": {"enabled": True}} if self.sparse else {}),
                **kwargs,
            )
        self.index = self.pc.Index(self.index_name)

    # ----------------------
    # Writing
    # ----------------------
    def write_to_index(self, records: List[Dict[str, Any]], namespace: str="default", **kwargs):
        """
        Upsert records into the Pinecone index.
        Args:
            records (List[Dict]): Each record should have 'id', 'values' (dense), and optionally 'sparse_values' (sparse) and 'metadata'.
        """
        try:
            pinecone_records = []
            for r in records:
                if self.sparse and "sparse_values" in r:
                    pinecone_records.append({
                        "id": r["id"],
                        "values": r.get("values"),
                        "sparse_values": r["sparse_values"],
                        "metadata": r.get("metadata", {})
                    })
                else:
                    pinecone_records.append((r["id"], r["values"], r.get("metadata", {})))
            self.index.upsert(vectors=pinecone_records, namespace=namespace, **kwargs)
        except Exception as e:
            print(f"Error upserting records: {e}")
            raise e

    def reindex(self, namespace: str = "default", batch_size: int = 100, embedding_endpoint: str = None, model_name: str = None, nvidia_api_key: str = None, input_type: str = "passage", truncate: str = "END", **kwargs):
        """
        Re-embed all vectors in the index using the NVIDIA embedding service and upsert them back, overwriting the originals.
        Args:
            namespace (str): The namespace to reindex.
            batch_size (int): Number of vectors to process at a time.
            embedding_endpoint (str, optional): NVIDIA embedding endpoint.
            model_name (str, optional): Embedding model name.
            nvidia_api_key (str, optional): NVIDIA API key.
            input_type (str): Input type for embedding (default 'passage').
            truncate (str): Truncation strategy (default 'END').
            **kwargs: Additional arguments for embedding or upsert.
        """
        # List all vector IDs in the namespace
        next_token = None
        while True:
            list_response = self.index.list(namespace=namespace, limit=batch_size, pagination_token=next_token)
            ids = [v['id'] for v in list_response.get('vectors', [])]
            if not ids:
                break
            # Fetch the vectors
            fetch_response = self.index.fetch(ids=ids, namespace=namespace)
            vectors = []
            for vid in ids:
                vec = fetch_response['vectors'].get(vid)
                if vec:
                    # Reconstruct the record for embedding
                    # Assume the original content is in metadata['content']
                    record = {
                        'id': vid,
                        'metadata': vec.get('metadata', {})
                    }
                    vectors.append(record)
            # Re-embed and upsert
            self.embed_index_collection(
                vectors,
                batch_size=batch_size,
                embedding_endpoint=embedding_endpoint,
                model_name=model_name,
                nvidia_api_key=nvidia_api_key,
                input_type=input_type,
                truncate=truncate,
                namespace=namespace,
                **kwargs
            )
            # Pagination
            next_token = list_response.get('pagination', {}).get('next')
            if not next_token:
                break

    def update_metadata(self, record_id: str, metadata: Dict[str, Any], namespace: str = "default"):
        """
        Update the metadata for a specific record in the Pinecone index.
        Args:
            record_id (str): The ID of the record to update.
            metadata (Dict[str, Any]): The metadata to update.
            namespace (str): The namespace of the record.
        """
        try:
            self.index.update(id=record_id, set_metadata=metadata, namespace=namespace)
        except Exception as e:
            print(f"Error updating metadata for record {record_id}: {e}")
            raise e

    def run(self, records: List[Dict[str, Any]]):
        """
        Create index and write records in one step.
        """
        self.create_index()
        self.write_to_index(records)

    # ----------------------
    # Searching
    # ----------------------
    def retrieval(self, queries: List[Any], top_k: int = 5, namespace: str="default", **kwargs) -> List[Any]:
        """
        Query the Pinecone index for nearest neighbors.
        Args:
            queries (List): List of query vectors. Each query can be a dense vector or a dict with 'values' and 'sparse_values'.
            top_k (int): Number of results to return.
            namespace (str): Namespace to query.
        Returns:
            List of query results.
        """
        results = []
        for q in queries:
            if self.sparse and isinstance(q, dict) and "sparse_values" in q:
                res = self.index.query(
                    vector=q.get("values"),
                    sparse_vector=q["sparse_values"],
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace,
                    **kwargs
                )
            else:
                res = self.index.query(
                    vector=q if not isinstance(q, dict) else q.get("values"),
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace,
                    **kwargs
                )
            results.append(res)
        return results

    def reranked_retrieval(
        self,
        queries: list,
        top_k: int = 5,
        namespace: str = "default",
        nv_reranker_endpoint: str = None,
        nv_reranker_model_name: str = None,
        nv_reranker_nvidia_api_key: str = None,
        truncate: str = "END",
        rerank_topk: int = 5,
        **kwargs
    ):
        """
        Retrieve results from Pinecone and rerank them using NVIDIA's reranker.
        """
        from nv_ingest_client.util.vdb.milvus import nv_rerank

        results = self.retrieval(queries, top_k=top_k, namespace=namespace, **kwargs)
        reranked_results = []
        for query, result in zip(queries, results):
            candidates = [match for match in result['matches']]
            reranked = nv_rerank(
                query=query,
                candidates=candidates,
                reranker_endpoint=nv_reranker_endpoint,
                model_name=nv_reranker_model_name,
                nvidia_api_key=nv_reranker_nvidia_api_key,
                truncate=truncate,
                topk=rerank_topk,
            )
            reranked_results.append(reranked)
        return reranked_results

    # ----------------------
    # Fetching
    # ----------------------
    def fetch_vector(self, vector_id: str, namespace: str = "default"):
        """
        Fetch a vector and its metadata from the index by its ID.
        Args:
            vector_id (str): The ID of the vector to fetch.
            namespace (str): The namespace to search in.
        Returns:
            dict: The vector data and metadata, or None if not found.
        """
        try:
            response = self.index.fetch(ids=[vector_id], namespace=namespace)
            return response
        except Exception as e:
            print(f"Error fetching vector {vector_id}: {e}")
            raise e

    # ----------------------
    # Imports
    # ----------------------
    def bulk_import(self, uri: str, error_mode: str = "CONTINUE", integration_id: str = None):
        """
        Start an asynchronous bulk import of vectors from object storage (e.g., S3) into the index.
        Args:
            uri (str): The S3 URI (e.g., 's3://bucket/path/to/dir').
            error_mode (str): Error handling mode, either 'CONTINUE' or 'ABORT'.
            integration_id (str, optional): Integration ID for private buckets.
        Returns:
            dict: The response from Pinecone, including the operation_id.
        Reference: https://docs.pinecone.io/reference/api/2025-04/data-plane/start_import
        """
        try:
            from pinecone import ImportErrorMode
            mode = getattr(ImportErrorMode, error_mode.upper(), ImportErrorMode.CONTINUE)
            kwargs = {"uri": uri, "error_mode": mode}
            if integration_id:
                kwargs["integration_id"] = integration_id
            response = self.index.start_import(**kwargs)
            return response
        except Exception as e:
            print(f"Error starting bulk import: {e}")
            raise e

    def list_imports(self):
        """
        List all ongoing and completed import operations for the index.
        Returns:
            dict: The response from Pinecone listing all imports.
        """
        try:
            return self.index.list_imports()
        except Exception as e:
            print(f"Error listing imports: {e}")
            raise e

    def describe_import(self, operation_id: str):
        """
        Describe a specific import operation by its operation_id.
        Args:
            operation_id (str): The ID of the import operation to describe.
        Returns:
            dict: The response from Pinecone describing the import.
        """
        try:
            return self.index.describe_import(operation_id=operation_id)
        except Exception as e:
            print(f"Error describing import {operation_id}: {e}")
            raise e

    def cancel_import(self, operation_id: str):
        """
        Cancel an ongoing import operation by its operation_id.
        Args:
            operation_id (str): The ID of the import operation to cancel.
        Returns:
            dict: The response from Pinecone after attempting to cancel the import.
        """
        try:
            return self.index.cancel_import(operation_id=operation_id)
        except Exception as e:
            print(f"Error canceling import {operation_id}: {e}")
            raise e

    # ----------------------
    # Backups & Restore
    # ----------------------
    def create_backup(self, backup_name: str, description: str = None, tags: dict = None):
        """
        Create a backup of the current index.
        Args:
            backup_name (str): Name for the backup.
            description (str, optional): Description for the backup.
            tags (dict, optional): Tags for the backup.
        Returns:
            dict: The backup creation response.
        Reference: https://docs.pinecone.io/reference/api/2025-04/control-plane/create_backup
        """
        try:
            return self.pc.create_backup(
                index_name=self.index_name,
                backup_name=backup_name,
                description=description,
                tags=tags or {}
            )
        except Exception as e:
            print(f"Error creating backup: {e}")
            raise e

    def list_backups(self):
        """
        List all backups for all indexes in the project.
        Returns:
            dict: The response listing all backups.
        """
        try:
            return self.pc.list_backups()
        except Exception as e:
            print(f"Error listing all backups: {e}")
            raise e

    def list_index_backups(self):
        """
        List all backups for this index.
        Returns:
            dict: The response listing backups for this index.
        """
        try:
            return self.pc.list_backups(index_name=self.index_name)
        except Exception as e:
            print(f"Error listing backups for index {self.index_name}: {e}")
            raise e

    def describe_backup(self, backup_name: str):
        """
        Describe a backup by name.
        Args:
            backup_name (str): The name of the backup to describe.
        Returns:
            dict: The backup description.
        """
        try:
            return self.pc.describe_backup(backup_name=backup_name)
        except Exception as e:
            print(f"Error describing backup {backup_name}: {e}")
            raise e

    def delete_backup(self, backup_name: str):
        """
        Delete a backup by name.
        Args:
            backup_name (str): The name of the backup to delete.
        Returns:
            dict: The response from Pinecone after deletion.
        """
        try:
            return self.pc.delete_backup(backup_name=backup_name)
        except Exception as e:
            print(f"Error deleting backup {backup_name}: {e}")
            raise e

    def create_index_from_backup(self, new_index_name: str, backup_name: str, spec: dict = None):
        """
        Create a new index from a backup.
        Args:
            new_index_name (str): Name for the new index.
            backup_name (str): Name of the backup to restore from.
            spec (dict, optional): Index spec (cloud, region, etc.).
        Returns:
            dict: The response from Pinecone.
        """
        try:
            return self.pc.create_index_from_backup(
                index_name=new_index_name,
                backup_name=backup_name,
                spec=spec
            )
        except Exception as e:
            print(f"Error creating index from backup {backup_name}: {e}")
            raise e

    def list_restore_jobs(self):
        """
        List all restore jobs in the project.
        Returns:
            dict: The response listing restore jobs.
        """
        try:
            return self.pc.list_restore_jobs()
        except Exception as e:
            print(f"Error listing restore jobs: {e}")
            raise e

    def describe_restore_job(self, restore_job_id: str):
        """
        Describe a restore job by its ID.
        Args:
            restore_job_id (str): The ID of the restore job.
        Returns:
            dict: The restore job description.
        """
        try:
            return self.pc.describe_restore_job(restore_job_id=restore_job_id)
        except Exception as e:
            print(f"Error describing restore job {restore_job_id}: {e}")
            raise e

    # ----------------------
    # Embedding
    # ----------------------
    def embed_index_collection(
        self,
        data,
        batch_size: int = 256,
        embedding_endpoint: str = None,
        model_name: str = None,
        nvidia_api_key: str = None,
        input_type: str = "passage",
        truncate: str = "END",
        namespace: str = "default",
        **kwargs,
    ):
        """
        Embed records using NVIDIA's embedding microservice and upsert them into Pinecone.
        Args:
            data (list): List of records to embed and upsert. Each record must have 'metadata' with 'content'.
            batch_size (int): Batch size for embedding requests.
            embedding_endpoint (str, optional): NVIDIA embedding endpoint.
            model_name (str, optional): Embedding model name.
            nvidia_api_key (str, optional): NVIDIA API key.
            input_type (str): Input type for embedding (default 'passage').
            truncate (str): Truncation strategy (default 'END').
            namespace (str): Pinecone namespace to upsert into.
            **kwargs: Additional arguments for embedding or upsert.
        """
        from nv_ingest_client.util.transport import infer_microservice
        import math
        if not data or len(data) == 0:
            return
        # Batch processing
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            # Infer embeddings
            embeddings = infer_microservice(
                batch,
                model_name=model_name,
                embedding_endpoint=embedding_endpoint,
                nvidia_api_key=nvidia_api_key,
                input_type=input_type,
                truncate=truncate,
                batch_size=batch_size,
                grpc=False,
            )
            # Attach embeddings to records
            for record, emb in zip(batch, embeddings):
                record["values"] = emb
            # Upsert to Pinecone
            self.write_to_index(batch, namespace=namespace, **kwargs)
