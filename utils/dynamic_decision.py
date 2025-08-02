from functools import lru_cache
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

@lru_cache()
def get_embed_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

class DynamicDecisionEngine:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-8b-8192"
        )

    def query(self, user_query, context_chunks):
        context_str = "\n".join(context_chunks)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using only the given context. Don't make up answers."),
            ("human", "Context:\n{context}\n\nQuestion: {query}")
        ])

        chain = prompt | self.llm
        response = chain.invoke({"context": context_str, "query": user_query})
        return response.content
