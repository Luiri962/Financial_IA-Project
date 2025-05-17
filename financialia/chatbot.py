import pandas as pd
import nltk
from nltk.corpus import wordnet
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

# Descargar recursos necesarios de nltk
nltk.download('punkt', quiet=False)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab')

# Para cargar el dataset que esta en Excel
def cargar_dataset():
    try:
  
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construir la ruta al archivo Excel
        excel_path = os.path.join(script_dir, "Preguntas_chatbot_financiero.xlsx")
        
        # Cargar el archivo Excel
        df = pd.read_excel(excel_path)
        
       # Solo por saber si esta realizando bien la carga 
        required_columns = ['Categoria', 'Pregunta', 'Respuesta']
        
        # Normalizar nombres de columnas (ignorar mayúsculas/minúsculas)
        df.columns = [col.lower() for col in df.columns]
        required_columns_lower = [col.lower() for col in required_columns]
        
        # Para Comprobar si existen todas las columnas necesarias
        for col in required_columns_lower:
            if col not in df.columns:
                print(f"Error: La columna '{col}' no existe en el archivo Excel.")
                return None
        
        print(f"Dataset cargado correctamente con {len(df)} preguntas.")
        return df
    
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo Excel 'Preguntas_chatbot_financiero.xlsx'")
        print(f"Asegúrate de que el archivo está en la misma carpeta que este script.")
        return None
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None

# Función para obtener sinónimos de una palabra en español, si quieres Juan le puedes agregregar otros que veas necesario
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word, lang='spa'):
        for lemma in syn.lemmas(lang='spa'):
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

# Función para expandir una pregunta con sinónimos
def expand_query_with_synonyms(query):
    tokens = nltk.word_tokenize(query.lower())
    expanded_query = query.lower()
    
    # sinonimós
    financial_synonyms = {
        'mercado': ['bolsa', 'trading', 'finanzas', 'valores'],
        'predicción': ['pronóstico', 'estimación', 'previsión', 'proyección'],
        'retorno': ['rendimiento', 'ganancia', 'beneficio', 'utilidad'],
        'compañías': ['empresas', 'corporaciones', 'firmas', 'negocios'],
        'alcista': ['alza', 'subida', 'crecimiento', 'aumento'],
        'bajista': ['baja', 'caída', 'descenso', 'disminución'],
        'invertir': ['inversión', 'colocar dinero', 'especular', 'comprar'],
        'hola': [ 'qué tal', 'buena tarde', 'buenos días','buenas noches','hi','hello','oe'],
        'qué tal': ['hola'],
        'buenas tardes': ['hola'],
        'buenos días': ['hola'],
        'buenas noches': ['hola'],
        'hi': ['hola'],
        'hello': ['hola'],
        'oe': ['hola'],
    }
    
    # Añadir sinónimos del contexto financiero
    for token in tokens:
        token = token.lower()
        # Usar sinónimos predefinidos si existen
        if token in financial_synonyms:
            for synonym in financial_synonyms[token]:
                if synonym not in expanded_query:
                    expanded_query += " " + synonym
        # Usar WordNet para otros términos
        elif len(token) > 0:  # Solo considerar palabras más largas que 3 caracteres
            synset_synonyms = get_synonyms(token)
            for synonym in synset_synonyms:
                if synonym != token and synonym not in expanded_query:
                    expanded_query += " " + synonym
    
    return expanded_query

# Función para encontrar la pregunta más similar, acá utilizo similitud coseno
def find_best_match(user_query, questions):
    # Normalizar y limpiar la consulta
    user_query = re.sub(r'[¿?¡!.,;:()]', '', user_query.lower())
    
    # Expandir la consulta con sinónimos
    expanded_query = expand_query_with_synonyms(user_query)
    
    # Crear vectorizador TF-IDF
    vectorizer = TfidfVectorizer()
    
    # Normalizar y limpiar las preguntas del dataset
    processed_questions = [re.sub(r'[¿?¡!.,;:()]', '', q.lower()) for q in questions]
    
    all_questions = processed_questions + [expanded_query]
    
    # Transformar todas las preguntas y la consulta expandida
    tfidf_matrix = vectorizer.fit_transform(all_questions)
    
    # Se calcula la probabilidad coseno 
    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
    
    # Obtener el índice de la pregunta más similar
    best_match_idx = np.argmax(cosine_similarities)
    similarity_score = cosine_similarities[best_match_idx]
    
    # Si la similitud es baja, consideramos que no hay coincidencia
    if similarity_score < 0.24:
        return None, similarity_score
    
    return best_match_idx, similarity_score

# Función para verificar si la pregunta es sobre el retorno esperado del S&P500
def is_sp500_question(pregunta):
    # Convertir a minúsculas y eliminar signos de puntuación
    pregunta_clean = re.sub(r'[¿?¡!.,;:()]', '', pregunta.lower())
    
    # Palabras clave que deben estar presentes
    keywords_sp500 = ["sp500", "s&p500", "s&p 500", "sp 500"]
    keywords_retorno = ["retorno", "rendimiento", "ganancia", "beneficio"]
    
    # Verificar si contiene alguna palabra clave de S&P500
    contains_sp500 = any(keyword in pregunta_clean.replace(" ", "") for keyword in keywords_sp500)
    
    # Verificar si contiene alguna palabra clave de retorno
    contains_retorno = any(keyword in pregunta_clean for keyword in keywords_retorno)
    
    # Debe contener ambos tipos de palabras clave
    return contains_sp500 and contains_retorno

# Función principal del chatbot
def chat_finanzas():
    print("¡Bienvenido al Chatbot Financiero!")  #Le podemos cambiar a como quiera que salude el chat
    print("Escribe 'salir' para terminar la conversación.")
    
    # Cargar dataset desde Excel
    df = cargar_dataset()
    
    # Si no se pudo cargar el dataset, salir
    if df is None:
        print("No se puede iniciar el chatbot sin un dataset válido.")
        return
    
    # Flag para controlar si estamos en el flujo de predicción de S&P500
    waiting_for_days = False
    
    while True:
        # Si estamos esperando el número de días para la predicción
        if waiting_for_days:
            user_input = input("\nIngresa el número de días para la predicción (1-7): ").strip()
            
            # Verificar si el input es un número entre 1 y 7
            if user_input.isdigit() and 1 <= int(user_input) <= 7:
                days = int(user_input)
                print(f"Chatbot: Llamando al modelo para predecir el retorno del S&P500 para los próximos {days} días...")
                waiting_for_days = False
            else:
                print("Chatbot: Por favor, ingresa un número válido entre 1 y 7.")
                continue
        else:
            user_input = input("\nTú: ").strip()
            
            if not user_input:
                print("Chatbot: Por favor, haz una pregunta.")
                continue
                
            if user_input.lower() in ['salir', 'adios', 'chao', 'hasta luego', 'exit', 'quit']:
                print("Chatbot: ¡Hasta luego! Espero haberte ayudado.")
                break
            
            # Aquí miramos el dataset y miramos la mejor considencia 
            best_match_idx, similarity = find_best_match(user_input, df['pregunta'])
            
            if best_match_idx is None:
                print("Chatbot: Lo siento, no entiendo tu pregunta. ¿Podrías reformularla?")
                continue
            
            # Obtener la categoría, pregunta y respuesta correspondiente
            categoria = df.iloc[best_match_idx]['categoria']
            pregunta = df.iloc[best_match_idx]['pregunta']
            respuesta = df.iloc[best_match_idx]['respuesta']
            
            print(f"[Debug] Coincidencia encontrada: '{pregunta}' (Score: {similarity:.2f})")
            
            # Verificar si es la pregunta específica sobre S&P500
            if is_sp500_question(pregunta):
                print(f"[Debug] Detectada pregunta sobre S&P500: '{pregunta}'")
                print("Chatbot: Para calcular el retorno esperado del S&P500, necesito saber para cuántos días deseas la predicción.")
                waiting_for_days = True
            elif categoria.lower() == 'predicción':
                print("Chatbot: Llamando modelo...")
            else:  # Asumimos que es informativo
                print(f"Chatbot: {respuesta}")

# Punto de entrada principal
if __name__ == "__main__":
    chat_finanzas()