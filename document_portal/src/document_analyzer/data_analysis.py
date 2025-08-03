import os
import sys
from dotenv import load_dotenv
import pydantic
from utils.model_loader import Model_Loader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from models.model import *
from langchain.output_parsers import OutputFixingParser, JsonOutputToolsParser
from prompt.prompt_library import *

class DocumentAnalyzer:
    """
    Analyze documents using a pretrained model.
    Automatically logs all actions and supports session-based organization.
    """
    
    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.loader = Model_Loader()
            self.llm = self.loader.load_llm()
            
            self.parser = JsonOutputToolsParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            self.prompt = prompt
            self.log.info("DocumentAnalyzer initialized successfully")
        except Exception as e:
            self.log.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DocumentPortalException("Error initializing DocumentAnalyzer", sys) 
            
        
    
    def analyze_document(self):
        pass
    
    
