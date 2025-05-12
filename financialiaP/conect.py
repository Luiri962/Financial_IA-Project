import psycopg2

# Conectar a PostgreSQL
conn = psycopg2.connect(
    dbname="predicciones_db", 
    user="Postgres", 
    password="SP500", 
    host="localhost", 
    port="5432"
)

# Crear un cursor para interactuar con la base de datos
cursor = conn.cursor()

# Ejecutar una consulta
cursor.execute("SELECT version();")
db_version = cursor.fetchone()
print(f"Conexión exitosa a PostgreSQL, versión: {db_version}")

# Cerrar la conexión
cursor.close()
conn.close()
