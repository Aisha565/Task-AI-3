from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTORSTORE_PATH = "vectorstore/faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"


class DenseRetriever:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME
        )

        self.vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def search(self, query, k=5):

        docs = self.vectorstore.similarity_search(query, k=k)

        results = []

        for i, doc in enumerate(docs):

            results.append({
                "content": doc.page_content,
                "score": 1 / (i + 1),
                "source": "dense"
            })

        return results