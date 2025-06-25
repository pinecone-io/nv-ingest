from nv_ingest_client.util.vdb.adt_vdb import VDB
from nv_ingest_client.util.vdb.milvus import Milvus
from nv_ingest_client.util.vdb.pinecone import Pinecone

available_vdb_ops = {
    "milvus": Milvus,
    "pinecone": Pinecone,
}
__all__ = ["VDB", "available_vdb_ops"]
