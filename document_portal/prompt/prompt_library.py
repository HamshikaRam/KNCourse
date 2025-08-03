from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a highly trained assistant capable of analyzing documents and summarize them.
Return only valid JSON data matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")