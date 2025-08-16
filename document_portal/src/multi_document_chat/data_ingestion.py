import sys

from requests import session
from exception.custom_exception import DocumentPortalException
from tokenize import Single
import uuid
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from logger.custom_logger import CustomLogger
from utils.model_loader import Model_Loader
from datetime import datetime, timezone


class DocumentIngestor:
    SUPPORTED_FILE_EXTN = {'.pdf','.docx','.txt','.md'}
    def __init__(self, temp_dir: str = "data/multi_doc_chat", faiss_dir: str = "faiss_index", session_id: str|None = None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            
            #base paths
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok = True)
            self.faiss_dir = Path(faiss_dir)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)
            
            #sessionized directories
            self.session_id = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.session_temp_dir = self.temp_dir / self.session_id
            self.session_faiss_dir = self.faiss_dir / self.session_id
            self.session_temp_dir.mkdir(parents=True, exist_ok=True)
            self.session_faiss_dir.mkdir(parents=True, exist_ok=True)

            self.model_loader = Model_Loader()
            self.log.info(
                "Document Ingestor Initialized",
                temp_base = str(self.temp_dir),
                faiss_base = str(self.faiss_dir),
                session_id = self.session_id,
                temp_path = str(self.session_temp_dir),
                faiss_path = str(self.session_faiss_dir)
                )
            
        except Exception as e:
            self.log.error("Failed to initialize DocumentIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in DocumentIngestor", sys)
    
    def ingest_files(self, uploaded_files):
        try:
            documents = []
            
            for uploaded_file in uploaded_files:
                ext = Path(uploaded_file.name).suffix.lower()
                if ext not in self.SUPPORTED_FILE_EXTN:
                    self.log.warning("Unsupported file skipped",filename=uploaded_file.name)
                    continue
                
                unique_filename = f"{uuid.uuid4().hex[:8]}.{ext}"
                temp_path = self.session_temp_dir / unique_filename
                
                with open(temp_path, "wb") as f_out:
                    f_out.write(uploaded_file.read())
                self.log.info("FIle saved for ingestion", filename = uploaded_file.name, saved_as = str(temp_path), session_id = self.session_id)
                
                if ext == ".pdf":
                    loader = PyPDFLoader(str(temp_path))
                elif ext == ".docx":
                    loader = Docx2txtLoader(str(temp_path))
                elif ext == ".txt":
                    loader = TextLoader(str(temp_path), encoding = "utf-8")
                else:
                    self.log.warning("Unsupported file type encountered", filename = uploaded_file.name)
                    continue
                
                
                docs = loader.load()
                documents.extend(docs)
                
            if not documents:
                raise DocumentPortalException("No valid documents loaded", sys)
            
            self.log.info("All documents loaded", total_docs=len(documents),session_id = self.session_id) 
            return self._create_retriever(documents)
                
        except Exception as e:
            self.log.error("Failed to ingest file for multi doc chat", error = str(e))
            raise DocumentPortalException("File ingestion error in DocumentIngestor", sys)
    
    def _create_retriever(self, documents):
        try:
            #can use different splitter, the option can be given in UI too
            splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 300)
            texts = splitter.split_documents(documents)
            self.log.info("Documents split into text", count=len(texts)) 
            
            embeddings = self.model_loader.load_embeddings()
            vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
            vectorstore.save_local(str(self.session_faiss_dir))
            self.log.info("FAISS index created and saved", faiss_path=str(self.session_faiss_dir))
            
            retriever = vectorstore.as_retriever(search_type = "similarity",search_kwargs ={"k":5})
            self.log.info("FAISS Retriver created and ready to use",session_id = self.session_id)
            return retriever
        except Exception as e:
            self.log.error("Failed to create retriever", error =str(e))
            raise DocumentPortalException("Retreiver creation error in DocumentIngestor", sys)