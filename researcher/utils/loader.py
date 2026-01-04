from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import os



def load_documents(folder_path: str) -> list[Document]:
    documents = []
    for file_name in os.listdir(folder_path):
        ext = os.path.splitext(file_name)[1].lower()
        file_path = os.path.join(folder_path, file_name)
        match ext:
            case ".txt" | ".md" | ".typ":
                documents.append(
                    Document(
                        page_content=open(file_path, "r").read(),
                        metadata={"source": file_name},
                    )
                )

            case ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)

    return documents