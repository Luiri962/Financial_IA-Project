from model.arquitectura import Model_LSTM
import os

def main():
    # Define las variables de entrada
    variables = ["^GSPC", "^DJI", "^IXIC", "^RUT", "PL=F", "UUP",
        "GC=F", "^GDAXI","SMCI", "BBD",
        "XRT", "XLK", "XLF", "SI=F",
        "AAPL", "GOOG", "AMZN", "MSFT", "TSLA", "NVDA"]
    start_date ='2015-01-01'
    target_variable = '^GSPC'  # Variable objetivo
    
    # Crear instancia del modelo
    model = Model_LSTM(
        variables=variables,
        start_date=start_date,
        target_variable=target_variable,
        look_back=24,
        future_periods=3
    )
    
    # Cargar y preprocesar datos
    data = model.load_data()
    data_scaled, dates, returns = model.preprocess_data()
    
    # Crear datasets
    X, y = model.create_dataset()
    X_train, X_test, y_train, y_test = model.split_data(X, y)
    
    # Construir y entrenar modelo
    model.build_model()
    history = model.train_model(X_train, y_train, X_test, y_test)
    
    # Evaluar modelo
    predictions, actual, rmse, mae, mape = model.evaluate_model(X_test, y_test)
    
    # Definir ruta completa para guardar el modelo
    model_dir = os.path.join(os.path.dirname(__file__), "model", "modelo")
    model_name = model.save_model(model_dir=model_dir)  
    
    # Realizar predicciones
    print("\nPredicciones futuras:")
    predictions, pred_retun = model.predict_future(periods=5)
    
    # Mostrar predicciones con precios y retornos
    print("\nFecha\t\tPrecio\t\tRetorno")
    print("-" * 50)
    for pred in predictions:
        print(f"{pred['date']}\t${pred['price']:.2f}\t{pred['return']:.2f}")

if __name__ == "__main__":
    main()
    
    
