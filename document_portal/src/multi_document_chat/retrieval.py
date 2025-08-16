import sys
import streamlit as st
import os

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

from utils.model_loader import Model_Loader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import PROMPT_REGISTRY
from models.model import PromptType



class ConversationalRag:
    def __init__(self, session_id:str, retriever=None):
        try:
            self.log = CustomLogger().get_logger()
            self.session_id = session_id
            self.llm = self._load_llm()
            self.contextualizeprompt = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qaprompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            
            if retriever is None:
                raise ValueError("Retriever cannot be none")
            
            self.retriever = retriever
            self._build_lcel_chain()
            self.log.info("Conversational RAG initialized", session_id = self.session_id)
            
        except Exception as e:
            self.log.error("Failed to initialize Conversational RAG", error = str(e))
            raise DocumentPortalException("Failed to initialise Conversation RAG", sys) 
           
    def load_retriever_from_faiss(self, index_path: str):
        try:
            embeddings = Model_Loader().load_embeddings()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"Faiss Index directory not found {index_path}")
            vectorstore = FAISS.load_local(
                embeddings,
                index_path,
                allow_dangerous_deserialization=True
            )
            self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})
            self.log.info("Faiss retriever loaded successfully", index_path=index_path, session_id = self.session_id)
            
            self._build_lcel_chain()
            return self.retriever
            
        except Exception as e:
            self.log.error("Failed to load retriever from FAISS", error = str(e))
            raise DocumentPortalException("Failed to load retriever from FAISS", sys)
    def invoke(self):
        try:
            pass
        except Exception as e:
            self.log.error("Failed to invoke Conversational RAG", error=str(e))
            raise DocumentPortalException("Failed to invoke Conversational RAG", sys)
    
    def _load_llm(self):
        try:
            llm = Model_Loader().load_llm()
            if not llm:
                raise ValueError("LLM is not loaded")
            self.log.info("LLM loaded successfully", session_id  = self.session_id)
            return llm
            
        except Exception as e:
            self.log.error("Failed to load llm", error =str(e))
            raise DocumentPortalException("Failed to load llm", sys)
    
    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)
    
    def _build_lcel_chain(self):
        try:
            
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                |self.llm
                |self.contextualizeprompt
                |StrOutputParser()
                
            )
            
            retrieve_docs = self.retriever | self._format_docs
            self.chain = (
                {
                    "context":retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history" : itemgetter("chat_history")
                }
                |self.llm
                |self.qaprompt
                |StrOutputParser()
            )
            
            self.log.info("LCEL chain built successfully",session_id = self.session_id)
        except Exception as e:
            self.log.info("Chain building error in RAG", error = str(e))
            raise DocumentPortalException("Chain building error in RAG", sys)
    
    