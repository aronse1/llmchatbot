from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# My local documents
documents = SimpleDirectoryReader("data").load_data()

# Embeddings model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Language model
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# Create index
index = VectorStoreIndex.from_documents(documents)

# Perform RAG query
query_engine = index.as_query_engine()
response = query_engine.query(
    "Was ist zur Abgabe der Bachelorarbeit notwendig?")
print(response)
