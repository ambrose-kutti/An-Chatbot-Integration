from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

class FastEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

db = Chroma(
    persist_directory="chroma_store",
    collection_name="csv_data",
    embedding_function=FastEmbedder()
)

print("Total documents present in the chrom db is:", db._collection.count())
