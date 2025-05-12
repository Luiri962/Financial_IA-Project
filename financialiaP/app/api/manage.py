from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import sys
import os  
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from model.arquitectura import Model_LSTM
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import joblib
from keras.models import load_model
from datetime import datetime

# Importar las nuevas funciones que usan SQL directo
from database.database import create_tables, save_predictions_to_db, get_predictions

app = FastAPI()

@app.on_event("startup")
async def startup():
    logger.info("Iniciando la aplicación...")
    try:
        # Crear tablas al iniciar la aplicación usando SQL directo
        await create_tables()
        logger.info("Inicialización completada")
    except Exception as e:
        logger.error(f"Error durante la inicialización: {e}")

class PredictionRequest(BaseModel):
    target_variable: str
    periods: int

# Cargar el modelo entrenado y escalador
def load_trained_model():
    try:
        file_path = 'model/model.h5'
        trained_model = load_model(file_path)
        scaler = joblib.load('model/scaler.pkl')
        
        with open('model/model_info.json', 'r') as f:
            model_info = json.load(f)

        model_instance = Model_LSTM(
            variables=model_info['variables'],
            target_variable=None,
            start_date="2025-01-01",
            look_back=24,
            future_periods=None
        )
        
        model_instance.model = trained_model
        model_instance.scaler = scaler
        
        data = model_instance.load_data()
        model_instance.preprocess_data(data) 
        
        return model_instance, data
    except Exception as e:
        logger.error(f"Error al cargar el modelo entrenado: {e}")
        raise

# Endpoint de predicción
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        target_variable = request.target_variable
        periods = request.periods
        
        logger.info(f"Realizando predicción para {target_variable} con {periods} periodos")
        
        # Cargar el modelo entrenado
        model_instance, data = load_trained_model()

        # Actualizar la variable objetivo
        model_instance.target_variable = target_variable
        
        # Realizar predicciones futuras
        future_predictions = model_instance.predict_future(periods=periods)
        
        # Formatear las predicciones
        predictions = [{"date": prediction['date'], "target_variable": prediction['target_variable'], "price": prediction['price'], "return": prediction['return']} for prediction in future_predictions]
        
        # Intentar crear la tabla de nuevo antes de guardar
        await create_tables()
        
        # Guardar predicciones usando SQL directo
        success = await save_predictions_to_db(predictions)
        
        if not success:
            logger.warning("No se pudieron guardar las predicciones en la base de datos, pero se devuelven los resultados")
            return {"warnings": "No se pudieron guardar las predicciones en la base de datos", "predictions": predictions}
        
        logger.info(f"Predicción exitosa para {target_variable}")
        return {"predictions": predictions}
    
    except Exception as e:
        logger.error(f"Error en el endpoint de predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error al realizar la predicción: {str(e)}")

# Endpoint para obtener predicciones guardadas
@app.get("/predictions/{target_variable}")
async def get_saved_predictions(target_variable: str):
    try:
        logger.info(f"Obteniendo predicciones para {target_variable}")
        predictions = await get_predictions(target_variable)
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Error al obtener predicciones: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener predicciones: {str(e)}")

# Endpoint para obtener todas las predicciones
@app.get("/predictions")
async def get_all_predictions():
    try:
        logger.info("Obteniendo todas las predicciones")
        predictions = await get_predictions()
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Error al obtener todas las predicciones: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener todas las predicciones: {str(e)}")

# Endpoint de verificación de salud
@app.get("/health")
async def health_check():
    return {"status": "OK", "message": "API funcionando correctamente"}