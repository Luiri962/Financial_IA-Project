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

app = FastAPI()

chatbot = ChatbotFinanzas()

templates = Jinja2Templates(directory="../interfaz/templates")


@app.post("/chatbot", response_class=JSONResponse)
async def chat(request: Request, user_input: str = Form(...)):
    try:
        data = chatbot.cargar_dataset()
        if data is None:
            return JSONResponse(content={"error": "Error al cargar el dataset"}, status_code=500)

        # Si la pregunta es sobre el retorno de S&P500 o alguna de las acciones
        if chatbot.is_sp500_question(user_input) or chatbot.is_action_question(user_input):
            return JSONResponse(content={"response": "¿Cuántos periodos deseas predecir?"})

        # Encontrar la mejor coincidencia en el dataset para otras preguntas
        best_match_idx, _ = chatbot.find_best_match(user_input, data['pregunta'])
        
        if best_match_idx is None:
            return JSONResponse(content={"response": "Lo siento, no tengo una respuesta para eso."}, status_code=400)
        
        respuesta = data.iloc[best_match_idx]['respuesta']
        return JSONResponse(content={"response": respuesta})

    except Exception as e:
        logger.error(f"Error en /chatbot: {e}")
        return JSONResponse(content={"response": "Hubo un error en el chatbot."}, status_code=500)

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