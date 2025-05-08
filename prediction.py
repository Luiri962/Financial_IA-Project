import os
import json
from tensorflow.keras.models import load_model
import joblib
from model.arquitectura import Model_LSTM  # Asegúrate de que el código de tu clase Model_LSTM esté bien importado

def load_trained_model(model_name, model_dir='model/modelo'):
    """Carga un modelo previamente entrenado y su scaler"""
    # Eliminar el sufijo '.h5' si se pasa al cargar el modelo
    model_name = model_name.replace('.h5', '')
    
    # Verificar que el archivo json exista
    json_path = os.path.join(model_dir, f"{model_name}_info.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontró el archivo de información del modelo en: {json_path}")
    
    # Cargar información del modelo
    with open(json_path, 'r') as f:
        model_info = json.load(f)
    
    # Crear una instancia del modelo con la información cargada
    model_instance = Model_LSTM(
        variables=model_info['variables'],
        target_variable=model_info['target_variable'],
        start_date=model_info['start_date'],
        look_back=model_info['look_back'],
        future_periods=model_info['future_periods']
    )
    
    # Cargar el modelo Keras (.h5) sin el sufijo '.h5'
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo del modelo en: {model_path}")
    
    model_instance.model = load_model(model_path)
    
    # Cargar el scaler
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No se encontró el archivo del escalador en: {scaler_path}")
    
    model_instance.scaler = joblib.load(scaler_path)
    
    # Cargar los datos y preprocesarlos
    data = model_instance.load_data()  # Cargar datos
    data_scaled, dates, returns = model_instance.preprocess_data(data)  # Preprocesar datos
    
    return model_instance, data

def predict_future_for_variable(model_name, target_variable, periods=5, model_dir='model/modelo'):
    """Cargar un modelo entrenado y predecir futuros retornos para una variable específica"""
    # Cargar el modelo entrenado
    model_instance, data = load_trained_model(model_name, model_dir)
    
    # Actualizar la variable objetivo
    model_instance.target_variable = target_variable
    
    # Realizar predicciones futuras
    future_predictions = model_instance.predict_future(periods=periods)
    
    # Mostrar las predicciones futuras
    print(f"Predicciones futuras para {target_variable}:")
    for prediction in future_predictions:
        print(f"Fecha: {prediction['date']}, Retorno: {prediction['return']}, Precio: {prediction['price']}")

if __name__ == "__main__":
    # Nombre del modelo entrenado que deseas cargar
    model_name = "^GSPC_model"  # Asegúrate de usar el nombre correcto del modelo guardado
    
    # Definir la variable objetivo para predecir (por ejemplo, AAPL, GOOGL, TSLA, etc.)
    target_variable =' AAPL'

    # Número de periodos para los cuales deseas predecir (por ejemplo, 10 días)
    periods = 3  # Cambia esto por el número de días que desees predecir

    # Llamada a la función para cargar el modelo y hacer predicciones
    predict_future_for_variable(model_name, target_variable, periods)
