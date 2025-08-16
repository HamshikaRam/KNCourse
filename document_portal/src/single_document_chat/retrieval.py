import sys
import streamlit as st
import os
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
    def __init__(self, session_id: str, retriever)-> None:
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.retriever = retriever
            self.llm = self._load_llm()
            self.contextualize_prompt = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            self.history_aware_retriever = create_history_aware_retriever(self.llm, self.retriever, self.contextualize_prompt)
            self.log.info("Created history aware retriver", session_id = self.session_id)
            self.qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
            self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.qa_chain)
            self.log.info("Created RAG chain", session_id = self.session_id)

            self.chain = RunnableWithMessageHistory(
                self.rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            self.log.info("Created RunnableWithMessageHistory", session_id=self.session_id)
        except Exception as e:
            self.log.error("Error initializing ConversationalRAG", error=str(e))
            raise DocumentPortalException("Error initializing ConversationalRAG", sys)
        
    def _load_llm(self):
        try:
            llm = Model_Loader().load_llm()
            self.log.info("LLM loaded successfully", class_name = llm.__class__.__name__)
            return llm
            # llm = ModelLoader().load_model()
        except Exception as e:
            self.log.error("Error loading LLM", error=str(e))
            raise DocumentPortalException("Error loading LLM", sys)
        
    def _get_session_history(self, session_id: str):
        try:
            if "store" not in st.session_state:
                st.session_state.store = {}

            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
                self.log.info("New chat session history created", session_id=session_id)

            return st.session_state.store[session_id]
        except Exception as e:
            self.log.error("Error getting session history", error=str(e))
            raise DocumentPortalException("Error getting session history", sys)
        
    def load_retriever_from_faiss(self, index_path: str):
        try:
            embeddings = Model_Loader().load_embeddings()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found {index_path}")
            
            vectorstore = FAISS.load_local(index_path, embeddings)
            self.log.info("Loaded retriever from FAISS index", index_path=index_path)
            return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})
        except Exception as e:
            self.log.error("Error loading the retriever from faiss vector db", error=str(e))
            raise DocumentPortalException("Error retriving from faiss vector db",sys)
        
    def invoke(self, user_input: str)-> str:
        try:
            print(self.session_id)
            response = self.chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}}
            )
            answer = response.get("answer","No answer")
            if not answer:
                self.log.warning("Empty answer received", session_id = self.session_id)
            self.log.info("Chain invoked successfully", session_id = self.session_id, user_input= user_input, answer_preview=answer[:150])
            return answer
        
        except Exception as e:
            self.log.error("Failed to load Conversational RAG", error=str(e), session_id = self.session_id)
            raise DocumentPortalException("Failed to load Conversational RAG", sys)
        
        