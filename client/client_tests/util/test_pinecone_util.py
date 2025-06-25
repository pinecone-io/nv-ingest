import pytest
import unittest
from unittest.mock import MagicMock, patch
from nv_ingest_client.util.vdb.pinecone import PineconeVDB
from nv_ingest_client.util.vdb.adt_vdb import VDB

@pytest.fixture
def pinecone_config():
    return {
        "api_key": "fake-key",
        "cloud": "aws",
        "region": "us-east-1",
        "index_name": "test-index",
        "dimension": 8,
        "metric": "cosine",
        "sparse": False,
    }

@pytest.fixture
def sample_records():
    return [
        {"id": "1", "values": [0.1] * 8, "metadata": {"content": "foo"}},
        {"id": "2", "values": [0.2] * 8, "metadata": {"content": "bar"}},
    ]

@pytest.fixture
def sample_queries():
    return [[0.1] * 8, [0.2] * 8]

class TestPineconeVDB(unittest.TestCase):
    @patch("nv_ingest_client.util.vdb.pinecone.Pinecone")
    def setUp(self, mock_pc):
        self.mock_pc = mock_pc
        self.mock_index = MagicMock()
        self.mock_pc.return_value.Index.return_value = self.mock_index
        self.vdb = PineconeVDB(
            api_key="fake-key",
            cloud="aws",
            region="us-east-1",
            index_name="test-index",
            dimension=8,
            metric="cosine",
            sparse=False,
        )

    def test_init(self):
        self.assertEqual(self.vdb.index_name, "test-index")
        self.assertEqual(self.vdb.dimension, 8)
        self.assertEqual(self.vdb.metric, "cosine")
        self.assertFalse(self.vdb.sparse)

    def test_create_index(self):
        self.vdb.create_index()
        self.mock_pc.return_value.create_index.assert_called()
        self.mock_pc.return_value.Index.assert_called_with("test-index")

    def test_write_to_index(self):
        records = [
            {"id": "1", "values": [0.1] * 8, "metadata": {"content": "foo"}},
            {"id": "2", "values": [0.2] * 8, "metadata": {"content": "bar"}},
        ]
        self.vdb.write_to_index(records)
        self.mock_index.upsert.assert_called()

    def test_retrieval(self):
        self.mock_index.query.return_value = {"matches": []}
        queries = [[0.1] * 8]
        results = self.vdb.retrieval(queries)
        self.mock_index.query.assert_called()
        self.assertIsInstance(results, list)

    @patch.object(PineconeVDB, "embed_index_collection")
    def test_reindex(self, mock_embed):
        # Simulate list and fetch returning two vectors
        self.mock_index.list.side_effect = [
            {"vectors": [{"id": "1"}, {"id": "2"}], "pagination": {"next": None}},
        ]
        self.mock_index.fetch.return_value = {
            "vectors": {
                "1": {"metadata": {"content": "foo"}},
                "2": {"metadata": {"content": "bar"}},
            }
        }
        self.vdb.reindex()
        mock_embed.assert_called()

    def test_run(self):
        with patch.object(self.vdb, "create_index") as mock_create, \
             patch.object(self.vdb, "write_to_index") as mock_write:
            records = [
                {"id": "1", "values": [0.1] * 8, "metadata": {"content": "foo"}},
            ]
            self.vdb.run(records)
            mock_create.assert_called()
            mock_write.assert_called()

if __name__ == "__main__":
    unittest.main() 