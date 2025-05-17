import os
import sys
import uvicorn
from fastapi.staticfiles import StaticFiles
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fastapi import  FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from services.model_service import ModelService
from utils.logger import logger
from fastapi.templating import Jinja2Templates


app = FastAPI()

templates = Jinja2Templates(directory="../interfaz/templates")

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