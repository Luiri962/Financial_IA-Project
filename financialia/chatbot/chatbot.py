# chatbot.py
import pandas as pd
import nltk
from nltk.corpus import wordnet
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

class ChatbotFinanzas:
    def __init__(self):
        pass

    def cargar_dataset(self):
        nltk.download('punkt', quiet=False)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt_tab')
        
        try:
            #script_dir = os.path.dirname(os.path.abspath(__file__))
            #excel_path = os.path.join(script_dir, file_path)
            excel_path = "../chatbot/Preguntas_Chatbot_Financiero.xlsx"
        
            df = pd.read_excel(excel_path)
        
            required_columns = ['Categoria', 'Pregunta', 'Respuesta']

            df.columns = [col.lower() for col in df.columns]
            required_columns_lower = [col.lower() for col in required_columns]
           
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

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word, lang='spa'):
            for lemma in syn.lemmas(lang='spa'):
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def expand_query_with_synonyms(self, query):
        tokens = nltk.word_tokenize(query.lower())
        expanded_query = query.lower()
    
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
        
        for token in tokens:
            token = token.lower()
            if token in financial_synonyms:
                for synonym in financial_synonyms[token]:
                    if synonym not in expanded_query:
                        expanded_query += " " + synonym
        
            elif len(token) > 0: 
                synset_synonyms = self.get_synonyms(token)
                for synonym in synset_synonyms:
                    if synonym != token and synonym not in expanded_query:
                        expanded_query += " " + synonym
        
        return expanded_query

    def find_best_match(self, user_query, questions):
        user_query = re.sub(r'[¿?¡!.,;:()]', '', user_query.lower())
        expanded_query = self.expand_query_with_synonyms(user_query)

        vectorizer = TfidfVectorizer()
        
        processed_questions = [re.sub(r'[¿?¡!.,;:()]', '', q.lower()) for q in questions]
        
        all_questions = processed_questions + [expanded_query]

        tfidf_matrix = vectorizer.fit_transform(all_questions)
        
        cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
        
        best_match_idx = np.argmax(cosine_similarities)
        similarity_score = cosine_similarities[best_match_idx]
        
        if similarity_score < 0.24:
            return None, similarity_score
        
        return best_match_idx, similarity_score

    def is_sp500_question(self, pregunta):
        pregunta_clean = re.sub(r'[¿?¡!.,;:()]', '', pregunta.lower())
        
        keywords_sp500 = ["sp500", "s&p500", "s&p 500", "sp 500"]
        keywords_retorno = ["retorno", "rendimiento", "ganancia", "beneficio"]
        
        contains_sp500 = any(keyword in pregunta_clean.replace(" ", "") for keyword in keywords_sp500)
        contains_retorno = any(keyword in pregunta_clean for keyword in keywords_retorno)
        
        return contains_sp500 and contains_retorno
    
    def is_action_question(self, pregunta):
        acciones = ["aapl", "goog", "amzn", "msft", "tsla"]
        pregunta_clean = re.sub(r'[¿?¡!.,;:()]', '', pregunta.lower())
        
        return any(action in pregunta_clean for action in acciones)
    
    def chat(self, user_input):
        if self.is_sp500_question(user_input):
            self.target_variable = "^GSPC"  # Definir el S&P500 como la variable a predecir
            return "¿Cuántos periodos deseas predecir?"
        
        if self.is_action_question(user_input):
            # Detectar cuál acción se menciona y asignarla
            if "apple" in user_input.lower():
                self.target_variable = "AAPL"
            elif "google" in user_input.lower():
                self.target_variable = "GOOG"
            elif "amazon" in user_input.lower():
                self.target_variable = "AMZN"
            elif "microsoft" in user_input.lower():
                self.target_variable = "MSFT"
            elif "tesla" in user_input.lower():
                self.target_variable = "TSLA"
            return f"¿Cuántos periodos deseas predecir para {self.target_variable}?"

        return "Lo siento, no tengo una respuesta para esa pregunta."
