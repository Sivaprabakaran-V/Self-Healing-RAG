import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

class MockEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.1 * i] * 1536 for i, _ in enumerate(texts)]
    def embed_query(self, text):
        return [0.1] * 1536

persist_dir = "./scratch_chroma_db"
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

embeddings = MockEmbeddings()
db = Chroma(collection_name="test_col", embedding_function=embeddings, persist_directory=persist_dir)

db.add_texts(texts=["hello world", "foo bar"], ids=["id1", "id2"])

# Get by ID and include embeddings
res = db.get(ids=["id1", "id2"], include=["embeddings", "documents"])
print("Keys in get response:", res.keys())
print("Embeddings list length:", len(res["embeddings"]))
print("Embedding dimension:", len(res["embeddings"][0]))

shutil.rmtree(persist_dir)
