import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda



class SimpleRAG:
    def __init__(self):
        self.embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore = None

        self.retrieval_chain = None
        self.qa_chain = None
        self.full_rag_chain = None

    def load_documents(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл документов не найден по пути: {path}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = [text[i:i + 500] for i in range(0, len(text), 400)]  # Перекрытие 100 символов

        self.vectorstore = FAISS.from_texts(chunks, self.embedding)
        print(f"Загружено {len(chunks)} чанков в FAISS.")

        self.retrieval_chain = self.vectorstore.as_retriever(
            search_kwargs={"k": 3})  # Получаем 3 наиболее релевантных документа

        template = """Используй только следующий контекст, чтобы ответить на вопрос.
        Если ты не можешь найти ответ в контексте, просто скажи, что не знаешь, не пытайся придумывать.
        Отвечай на русском языке.

        Контекст:
        {context}

        Вопрос:
        {question}

        Ответ:"""

        self.qa_prompt = ChatPromptTemplate.from_template(template)

        self.qa_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}  # Пропускаем context и question
                | self.qa_prompt
                | self.llm
                | StrOutputParser()  # Парсим вывод LLM в строку
        )

        self.full_rag_chain = (
                {"context": self.retrieval_chain, "question": RunnablePassthrough()}
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
        )

    def run(self, question: str) -> str:
        if not self.full_rag_chain:
            raise RuntimeError("RAG система не инициализирована. Загрузите документы.")

        result = self.full_rag_chain.invoke({"question": question})
        return result


app = FastAPI()
rag_system = SimpleRAG()


class Query(BaseModel):
    question: str


@app.on_event("startup")
async def startup_event():
    try:
        rag_system.load_documents("documents/data.txt")
        print("RAG система успешно загружена!")
    except FileNotFoundError as e:
        print(f"ОШИБКА: {e}")
        print("Пожалуйста, убедитесь, что файл 'documents/data.txt' существует и содержит данные.")
    except Exception as e:
        print(f"Неизвестная ошибка при загрузке RAG: {e}")


@app.post("/ask")
async def ask_question(query: Query):
    try:
        answer = rag_system.run(query.question)
        return {"question": query.question, "answer": answer}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Произошла ошибка при обработке запроса: {e}")