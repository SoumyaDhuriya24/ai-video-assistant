import os 
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : 'cpu'}
    )

def build_vector_store(transcript : str)->Chroma:
    print("Building vector Store")

    # Guard against empty or whitespace-only transcript
    clean_text = transcript.strip() if transcript else ""
    if not clean_text:
        clean_text = "No spoken dialogue or speech was detected in this recording."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    raw_chunks = splitter.split_text(clean_text)
    chunks = [c.strip() for c in raw_chunks if c.strip()]
    if not chunks:
        chunks = [clean_text]

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i,chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    # Reset any existing collection to avoid mixing meeting transcripts
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    except Exception:
        pass

    vector_store = Chroma.from_documents(
        documents= docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    return vector_store



def load_vector_store() ->Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function= embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def get_retriever(vector_store : Chroma, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )


