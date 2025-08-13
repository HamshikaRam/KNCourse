from math import e
import sys
import os
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from requests import session
from utils.model_loader import Model_Loader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import PROMPT_REGISTRY
from models.model import PromptType

class ConversationalRAG:
    def __init__(self, sessionid: str, retriever)-> None:
        try:
            self.log = CustomLogger().get_logger(__name__)
        except Exception as e:
            self.log.error("Error initializing ConversationalRAG", error=str(e))
            raise DocumentPortalException("Error initializing ConversationalRAG", sys)
        
    def _load_llm(self):
        try:
            pass
        except Exception as e:
            self.log.error("Error loading LLM", error=str(e))
            raise DocumentPortalException("Error loading LLM", sys)
        
    def _get_session_history(self):
        try:
            pass
        except Exception as e:
            self.log.error("Error getting session history", error=str(e))
            raise DocumentPortalException("Error getting session history", sys)
        
    def load_retriever_from_faiss(self):
        try:
            pass
        except Exception as e:
            self.log.error("Error loading the retriever from faiss vector db", error=str(e))
            raise DocumentPortalException("Error retriving from faiss vector db",sys)
        
    def invoke(self):
        try:
            pass
        except Exception as e:
            self.log.error("Failed to load Conversational RAG", error=str(e), session_id = self.session_id)
            raise DocumentPortalException("Failed to load Conversational RAG", sys)
        
        