from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import pandas as pd
import glob
import os
import sys

CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "csv_data"

# ---- EMBEDDING CLASS (must match app.py exactly) ----
class FastEmbedder:
    def __init__(self):
        print("Loading SentenceTransformer model...")
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Model loaded successfully.")
        except Exception as e:
            print("ERROR: Could not load MiniLM model.")
            print(str(e))
            sys.exit(1)

    def embed_documents(self, texts):
        if not texts:
            print("WARNING: embed_documents() received empty texts.")
            return []
        vectors = self.model.encode(texts).tolist()
        return vectors

    def embed_query(self, text):
        vector = self.model.encode([text])[0].tolist()
        return vector

embedder = FastEmbedder()

# ---- LOAD CSV FILES ----
csv_files = glob.glob("excel_tables/*.csv")

if not csv_files:
    print("ERROR: No CSV files found in /excel_tables/*.csv")
    sys.exit(1)

docs = []

for file in csv_files:
    print(f"Loading file: {file}")
    df = pd.read_csv(file)

    for idx, row in df.iterrows():
        content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(
            Document(
                page_content=content,
                metadata={"source_file": os.path.basename(file)}
            )
        )

print(f"Total documents prepared: {len(docs)}")

if len(docs) == 0:
    print("ERROR: No documents created from CSV files.")
    sys.exit(1)

# ---- TEST EMBEDDING ON FIRST DOCUMENT ----
test_vec = embedder.embed_query(docs[0].page_content)
if not test_vec:
    print("ERROR: Embedding returned EMPTY vector!")
    sys.exit(1)

print("Embedding test passed ✓")

# ---- BUILD CHROMA ----
print("Building Chroma DB...")

db = Chroma.from_documents(
    documents=docs,
    embedding=embedder,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME
)

print("Chroma store created successfully!")
print(f"Total embedded documents: {len(docs)}")
