import sys

from requests import session
from exception.custom_exception import DocumentPortalException
from tokenize import Single
import uuid
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
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
    
    def ingest_files(self):
        try:
            pass
        except Exception as e:
            self.log.error("Failed to ingest file for multi doc chat", error = str(e))
            raise DocumentPortalException("File ingestion error in DocumentIngestor", sys)
    
    def _create_retriever(self, documents):
        try:
            pass
        except Exception as e:
            self.log.error("Failed to create retriever", error =str(e))
            raise DocumentPortalException("Retreiver creation error in DocumentIngestor", sys)