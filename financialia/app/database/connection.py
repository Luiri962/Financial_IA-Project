import asyncpg
import logging
from datetime import datetime
from utils.logger import logger


class DatabaseConnection():
    
    def __init__(self):
        self.BD_USER = "postgres"
        self.BD_PASSWORD = "2025"
        self.DB_HOST = "localhost"
        self.DB_PORT = 5432
        self.DB_NAME = "predicciones_db"

    async def get_connection(self):
        try:
            conn = await asyncpg.connect(
                user=self.BD_USER,
                password=self.BD_PASSWORD,
                host=self.DB_HOST,
                port=self.DB_PORT,
                database=self.DB_NAME
            )
            logger.info("Conexión a la base de datos establecida.")
            return conn
        except Exception as e:
            logger.error(f'Error al conectar a la base de datos: {e}')
            raise
    
