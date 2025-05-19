import json
import joblib
import os
import sys
from keras.models import load_model
from database.connection import DatabaseConnection
from database.tables import create_table
from model.arquitectura import Model_LSTM
from utils.logger import logger
from datetime import datetime  

# Cargar el modelo
class ModelService:

    def __init__(self):
        super().__init__()

    def load_trained_model(self):
        try:
            model = load_model('model/model.h5')
            scaler = joblib.load('model/scaler.pkl')

            with open('model/model_info.json') as f:
                model_info = json.load(f)

            instance = Model_LSTM(
            variables=model_info['variables'],
            target_variable=None,
            start_date="2025-01-01",
            look_back=24,
            future_periods=None
            )
            instance.model = model
            instance.scaler = scaler
            data = instance.load_data()
            instance.preprocess_data(data)
            return instance, scaler

        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise
        
    async def save_predictions_to_db(self,predictions):
        try:
            await create_table()
            connection = await DatabaseConnection().get_connection()

            values = []
            for prediction in predictions:
                try:
                    date_value = prediction['date']
                    if isinstance(date_value, str):
                        date_value = datetime.strptime(date_value, '%Y-%m-%d')

                    values.append((
                        date_value,
                        prediction['target_variable'],
                        prediction['price'], 
                        prediction['return'] 
                    ))
                except KeyError as e:
                    logger.error(f"Falta la clave {e} en la predicción: {prediction}")
                    continue

            if values:
                async with connection.transaction():
                    await connection.executemany('''
                        INSERT INTO predictions(date, target_variable, predicted_price, predicted_return)
                        VALUES($1, $2, $3, $4)
                    ''', values)

                logger.info(f"Se guardaron {len(values)} predicciones en la base de datos")
            else:
                logger.warning("No se encontraron valores válidos para insertar")

            await connection.close()
            return True

        except Exception as e:
            logger.error(f"Error al guardar las predicciones: {e}")
            return False

    
    
    async def make_predictions(self,target_variable, periods):
        
        periods = int(periods) if isinstance(periods, str) else periods
        
        model_instance, _ = self.load_trained_model()
        
        model_instance.target_variable = target_variable
        future_predictions = model_instance.predict_future(periods=periods)

        predictions = [
            {"date": p['date'], "target_variable": p['target_variable'], "price": p['price'], "return": p['return']}
            for p in future_predictions
        ]
        
        await self.save_predictions_to_db(predictions)
        return predictions


    async def get_predictions(self, target_variable=None):
        try:
            connection = await DatabaseConnection().get_connection()

            if target_variable:

                rows = await connection.fetch('''
                    SELECT * FROM predictions 
                    WHERE target_variable = $1
                    ORDER BY date
                ''', target_variable)
            else:
                rows = await connection.fetch('''
                    SELECT * FROM predictions 
                    ORDER BY date
                ''')
            
            predictions = [
                {
                    "id": row['id'],
                    "date": row['date'].isoformat(),
                    "target_variable": row['target_variable'],
                    "price": row['predicted_price'],
                    "return": row['predicted_return']
                }
                for row in rows
            ]
            
            await connection.close()
            return predictions
        except Exception as e:
            logger.error(f"Error al obtener las predicciones: {e}")
            return [] 
        
            
 