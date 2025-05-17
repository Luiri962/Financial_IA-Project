import logging
from datetime import datetime
from database.connection import DatabaseConnection
from utils.logger import logger

async def create_table():
    try:
        logger.info("Intentando crear la tabla 'predicciones'...")
        connection = await DatabaseConnection().get_connection()
    
        table_exist = await connection.fetchval(''' 
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_name = 'predictions'
            )
        ''')
        
        if not table_exist:
            await connection.execute('''
                CREATE TABLE predictions (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    target_variable VARCHAR(50) NOT NULL,
                    predicted_price FLOAT NOT NULL,
                    predicted_return FLOAT NOT NULL
                )
            ''')
            logger.info("Tabla 'predictions' creada exitosamente.")
            
        await connection.close()
    except Exception as e:
        logger.error(f"Error al crear la tabla: {e}")
        
        

