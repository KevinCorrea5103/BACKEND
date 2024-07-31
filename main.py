from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="Promptior API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    message: str

# Cargar la API key desde la variable de entorno
api_key = os.getenv("OPENAI_API_KEY")

# Verificar si la API key se cargó correctamente
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set or empty")

llm = OpenAI(temperature=0.7, api_key=api_key)
embeddings = OpenAIEmbeddings(api_key=api_key)

# Función para procesar el PDF y crear el vectorstore
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return FAISS.from_documents(texts, embeddings)

# Procesar el PDF al inicio
pdf_path = "promptior.pdf"
vectorstore = process_pdf(pdf_path)

# Crear una instancia de memoria
memory = ConversationBufferMemory(input_key="input", memory_key="history")

# Crear el prompt template
prompt = PromptTemplate(
    input_variables=["history", "input", "context"],
    template="""Eres un asistente AI de Promptior, una empresa uruguaya especializada en inteligencia artificial. 
    Tu tarea es responder preguntas exclusivamente sobre Promptior y sus servicios de IA, 
    basándote en la información proporcionada en el contexto. 
    Si te preguntan sobre algo que no está relacionado con Promptior, amablemente redirige la conversación 
    hacia los servicios y capacidades de Promptior en el campo de la IA.

    Contexto relevante:
    {context}

    Historial de la conversación:
    {history}

    Humano: {input}
    AI:"""
)

# Crear la cadena LLM
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

@app.post("/chat")
async def chat(chat_input: ChatInput):
    relevant_docs = vectorstore.similarity_search(chat_input.message, k=2)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    response = chain.predict(input=chat_input.message, context=context)
    return {"response": response.strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)