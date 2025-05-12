import asyncpg
import asyncio
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de la conexión
DB_USER = "Postgres"
DB_PASSWORD = "2025"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "predicciones_db"

# Función para obtener una conexión a la base de datos
async def get_connection():
    try:
        conn = await asyncpg.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME
        )
        logger.info("Conexión a la base de datos establecida")
        return conn
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        raise

# Función para crear la tabla si no existe
async def create_tables():
    try:
        logger.info("Intentando crear las tablas...")
        connection = await get_connection()
        
        # Verificar si la tabla ya existe
        table_exists = await connection.fetchval('''
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_name = 'predictions'
            )
        ''')
        
        if table_exists:
            logger.info("La tabla 'predictions' ya existe")
        else:
            # Crear la tabla predictions si no existe
            await connection.execute('''
                CREATE TABLE predictions (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    target_variable VARCHAR(50) NOT NULL,
                    predicted_price FLOAT NOT NULL,
                    predicted_return FLOAT NOT NULL
                )
            ''')
            logger.info("Tabla 'predictions' creada exitosamente")

        # Verifica al final si la tabla se creó
        table_exists_after = await connection.fetchval('''
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_name = 'predictions'
            )
        ''')
        
        if table_exists_after:
            logger.info("Verificación final: La tabla 'predictions' existe")
        else:
            logger.error("¡La tabla 'predictions' no se pudo crear!")
            
        await connection.close()
    except Exception as e:
        logger.error(f"Error al crear la tabla: {e}")
        # No levantar la excepción para permitir que la aplicación continúe

# Función para guardar predicciones en la base de datos
async def save_predictions_to_db(predictions):
    try:
        # Intentar crear la tabla primero, en caso de que no exista
        await create_tables()
        
        connection = await get_connection()
        
        # Iniciar una transacción
        async with connection.transaction():
            # Preparar los datos para la inserción en lote
            values = []
            for prediction in predictions:
                date_value = prediction['date']
                if isinstance(date_value, str):
                    date_value = datetime.strptime(date_value, '%Y-%m-%d')
                
                # Insertar cada predicción
                await connection.execute('''
                    INSERT INTO predictions(date, target_variable, predicted_price, predicted_return)
                    VALUES($1, $2, $3, $4)
                ''', 
                date_value,
                prediction['target_variable'], 
                prediction['price'], 
                prediction['return'])
            
        logger.info(f"Se guardaron {len(predictions)} predicciones en la base de datos")
        await connection.close()
        return True
    except Exception as e:
        logger.error(f"Error al guardar las predicciones: {e}")
        return False

# Función para obtener predicciones de la base de datos
async def get_predictions(target_variable=None):
    try:
        connection = await get_connection()
        
        if target_variable:
            # Consulta con filtro por variable objetivo
            rows = await connection.fetch('''
                SELECT * FROM predictions 
                WHERE target_variable = $1
                ORDER BY date
            ''', target_variable)
        else:
            # Consulta sin filtro
            rows = await connection.fetch('''
                SELECT * FROM predictions 
                ORDER BY date
            ''')
        
        # Convertir los resultados a diccionarios
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