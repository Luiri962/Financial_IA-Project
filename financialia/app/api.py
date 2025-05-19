import os
import sys
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi import  FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from services.model_service import ModelService
from utils.logger import logger
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from chatbot.chatbot import ChatbotFinanzas
import uuid
import nltk

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

user_state = {} # Diccionario global para guardar el estado por cliente

app = FastAPI()

chatbot = ChatbotFinanzas()

templates = Jinja2Templates(directory="../interfaz/templates")


@app.post("/chatbot", response_class=JSONResponse)
async def chat(user_input: str = Form(...), client_id: str = Form(None)):
    global user_state

    if client_id is None:
        client_id = str(uuid.uuid4())

    state = user_state.get(client_id, {})

    dataset = chatbot.cargar_dataset()
    if dataset is None:
        return JSONResponse(content={"error": "Error al cargar el dataset"}, status_code=500)

    respuesta, nuevo_estado = await chatbot.handle_conversation(user_input, state, dataset)
    user_state[client_id] = nuevo_estado

    return JSONResponse(content={"response": respuesta, "client_id": client_id})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, target_variable: str = Form(...), periods: int = Form(...)):
    try:
        predictions = await ModelService().make_predictions(target_variable=target_variable, periods=periods)

        return templates.TemplateResponse("index.html", {"request": request, "predictions": predictions})

    except Exception as e:
        logger.error(f"Error en /predict: {e}")
        raise HTTPException(status_code=500, detail="Error al realizar la predicción")
    
    
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    predictions = await ModelService().get_predictions()
    return templates.TemplateResponse("index.html", {"request": request, "predictions": predictions})

app.mount("/static", StaticFiles(directory="../interfaz/static"), name="static")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)