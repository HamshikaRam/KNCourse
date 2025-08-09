from pydoc import doc
import sys
from pathlib import Path
import fitz
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

class DocumentIngestion:
    
    def __init__(self, base_dir:str = "data\\document_compare"):
        self.log = CustomLogger().get_logger(__name__)
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def delete_existing_files(self):
        try:
            if self.base_path.exists():
                for file in self.base_path.iterdir():
                    if file.is_file():
                        file.unlink()
                        self.log.info("Deleted file", path=str(file))
                self.log.info("All existing files deleted successfully", directory=str(self.base_path))
        except Exception as e:
            self.log.error("Error deleting existing files", error=str(e))
            raise DocumentPortalException("Error deleting existing files", sys)
    
    def save_uploaded_files(self, reference_file, actual_file): # ref file is v1 and act file is v2
        """
        Save uploaded files to the base directory.
        """
        try:
            self.delete_existing_files()
            self.log.info("Existing files deleted successfully")
            ref_path = self.base_path/reference_file.name
            act_path = self.base_path/actual_file.name
            if not reference_file.name.endswith(".pdf") or not actual_file.name.endswith(".pdf"):
                raise  ValueError("Only PDF files are allowed")
            
            with open(ref_path, "wb") as ref_f:
                ref_f.write(reference_file.getbuffer())
            with open(act_path, "wb") as act_f:
                act_f.write(actual_file.getbuffer())
                
            self.log.info("Files Saved", ref_file=str(ref_path), act_file=str(act_path))
            return ref_path, act_path
        except Exception as e:
            self.log.error("Error saving uploaded files", error=str(e))
            raise DocumentPortalException("Error saving uploaded files", sys)
    
    def read_pdf(self, pdf_path: Path)->str:
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError("PDF is encrypted and cannot be read: {pdf_path.name}")
                all_text = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    
                    if text.strip():
                        all_text.append(f"\n-- Page {page_num+1} -- \n {text}")
                self.log.info("PDF read successfully",file = str(pdf_path),pages =len(all_text))
                return "\n".join(all_text)
        except Exception as e:
            self.log.error("Error reading PDF file", error=str(e))
            raise DocumentPortalException("Error reading PDF file", sys)
    
    def combine_documents(self) -> str:
        """ Combines text from both reference and actual documents.
        """
        try:
            content_dict = {}
            doc_parts = []
            
            for filename in sorted(self.base_path.iterdir()):
                if filename.is_file() and filename.suffix.lower() == ".pdf":
                    content_dict[filename.name] = self.read_pdf(filename)
                    
            for filename, content in content_dict.items():
                doc_parts.append(f"Document: {filename}\n{content}")
                
            combined_text = "\n\n".join(doc_parts)
            self.log.info("Documents combined successfully", total_parts=len(doc_parts))
            
            return combined_text
        
        except Exception as e:
            self.log.error("Error combining documents", error=str(e))
            raise DocumentPortalException("Error combining documents", sys)
    