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
        self.chain = self.prompt | self.llm | self.parser 
        self.log.info("DocumentComparatorLLM initialized with model and parser successfully")
        
        
        
    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        """
        Compares 2 documents and returns structured comparison as result.
        """
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instructions": self.parser.get_format_instructions()
            }
            self.log.info("Starting document comparison", inputs=inputs)
            response = self.chain.invoke(inputs)
            self.log.info("Document comparison completed successfully", response=response)
            
            return self._format_response(response)
        except Exception as e:
            self.log.error("Error in document comparison", error=str(e))
            raise DocumentPortalException("Error in document comparison", sys)
        
    def _format_response(self, response_parsed: dict) -> pd.DataFrame:
        """
        Formats the response from the LLM into a DataFrame.
        """
        try:
            df = pd.DataFrame(response_parsed)
            self.log.info("Response formatted into DataFrame successfully", DataFrame=df)
            return df
        except Exception as e:
            self.log.error("Error formatting response into DF", error=str(e))
            raise DocumentPortalException("Error formatting response", sys) 


