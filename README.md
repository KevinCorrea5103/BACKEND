# Promptior AI System

## Frontend (React Application)
### Componente Chat
- Estado de mensajes
- Componente MessageList
  - Muestra los mensajes
- Componente MessageInput
  - Entrada del usuario

## HTTP Request (axios)

## Backend (FastAPI)
### Endpoint /chat
- Recibe el mensaje del usuario
### Vectorstore FAISS
- Creado al inicio de la aplicación
- Contiene embeddings del PDF de Promptior
### Búsqueda de similitud
- Encuentra contexto relevante basado en el mensaje del usuario
### LLMChain
- Prompt Template
  - Incluye contexto, historial y mensaje del usuario
- Memoria de conversación
### OpenAI API
- Genera respuesta basada en el prompt
### Respuesta generada

## HTTP Response

## Frontend (React Application)
- Actualiza el estado de mensajes con la respuesta del AI

## Flujo de datos
1. Usuario escribe mensaje
2. Componente Chat envía solicitud POST
3. Backend recibe solicitud
4. Búsqueda en Vectorstore FAISS
5. Creación de prompt
6. Envío a API de OpenAI
7. Generación de respuesta
8. Backend envía respuesta al frontend
9. Actualización del estado de mensajes
10. Actualización de MessageList

## Componentes clave
- PDF de Promptior
- Vectorstore FAISS
- LLMChain
- Memoria de conversación
- API de OpenAI*# PROMPTIOR-BACKEND



### para instalar el proyecto

## pip install -r requirements.txt



### para iniciar el proyecto 

 ## uvicorn main:app --reload    






![Diagrama de Funcionamiento del Sistema Promptior AI](./mapa.svg)