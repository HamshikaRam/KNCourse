import sys
from dotenv import load_dotenv
import pandas as pd
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from models.model import *
from prompt.prompt_library import PROMPT_REGISTRY
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from utils.model_loader import Model_Loader


class DocumentComparatorLLM:
    def __init__(self):
        load_dotenv()
        self.log = CustomLogger().get_logger(__name__)
        self.loader = Model_Loader()
        self.llm = self.loader.load_llm()
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser=OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.prompt = PROMPT_REGISTRY.get("document_comparison") #can also be called this way
        self.chain = self.prompt | self.llm | self.parser | self.fixing_parser
        self.log.info("DocumentComparatorLLM initialized with model and parser successfully")
        
        
        
    def compare_documents(self):
        try:
            pass
        except Exception as e:
            self.log.error("Error in document comparison", error=str(e))
            raise DocumentPortalException("Error in document comparison", sys)
    def _format_response(self):
        try:
            pass
        except Exception as e:
            self.log.error("Error formatting response into DF", error=str(e))
            raise DocumentPortalException("Error formatting response", sys)


