from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import time
import aiohttp
from bs4 import BeautifulSoup
import asyncio
from functools import lru_cache

load_dotenv()

app = FastAPI(title="Promptior API")



# CORS Configuration
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatInput(BaseModel):
    message: str

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set or empty")

llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(api_key=api_key)

# Optimizar procesamiento de PDF
@lru_cache(maxsize=1)
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return FAISS.from_documents(texts, embeddings)

pdf_path = os.getenv("PDF_PATH", "promptior.pdf")
vectorstore = process_pdf(pdf_path)

memory = ConversationBufferMemory(input_key="input", memory_key="history")

prompt = ChatPromptTemplate.from_template(
    """Eres un asistente AI de Promptior, una empresa uruguaya de IA. Responde sobre Promptior y sus servicios de IA basándote en el contexto y contenido web proporcionados. Redirige preguntas no relacionadas hacia los servicios de Promptior en IA.

    Contexto: {context}
    Web: {web_content}
    Historia: {history}
    Humano: {input}
    AI:"""
)

chain = (
    {
        "context": lambda x: vectorstore.similarity_search(x["input"], k=1)[0].page_content[:200],
        "web_content": lambda x: x["web_content"][:200],
        "input": lambda x: x["input"],
        "history": lambda x: memory.load_memory_variables({})["history"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

web_content_cache = {"content": "", "last_updated": 0}
async def retrieve_web_content(url):
    current_time = time.time()
    if current_time - web_content_cache["last_updated"] > 3600:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')
                    paragraphs = soup.find_all('p')
                    web_content_cache["content"] = " ".join([p.get_text() for p in paragraphs])[:500]
                    web_content_cache["last_updated"] = current_time
                else:
                    web_content_cache["content"] = "No se pudo obtener contenido web."
    return web_content_cache["content"]


@app.post("/chat")
async def chat(chat_input: ChatInput, background_tasks: BackgroundTasks):
    start_time = time.time()
    
    web_content = await retrieve_web_content("https://promptior.ai")
    
    response = await asyncio.to_thread(
        chain.invoke,
        {"input": chat_input.message, "web_content": web_content}
    )
    
    background_tasks.add_task(memory.save_context, {"input": chat_input.message}, {"output": response})
    
    end_time = time.time()
    print(f"Chat processing took {end_time - start_time:.2f} seconds")
    
    return {"response": response.strip()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=4)