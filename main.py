from fastapi import FastAPI
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

df = pd.read_csv('C:\Fabrizzio\HENRY CURSO\Project\HenryLabs.csv')
df['release_date'].fillna('', inplace=True)
df['release_date'] = df['release_date'].astype(str)

ml_df = pd.read_csv('VariablesML.csv')

@app.get('/')
def saludo(): 
    return 'Hola a todos'

@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    '''Se ingresa el mes y la función retorna la cantidad de películas que se estrenaron ese mes históricamente'''
    
    # Convertir el mes ingresado a minúsculas y eliminar espacios adicionales
    mes = mes.lower().strip()

    # Mapear el nombre del mes al formato numérico correspondiente (por ejemplo, 'enero' -> '01')
    meses_dict = {
        'enero': '01',
        'febrero': '02',
        'marzo': '03',
        'abril': '04',
        'mayo': '05',
        'junio': '06',
        'julio': '07',
        'agosto': '08',
        'septiembre': '09',
        'octubre': '10',
        'noviembre': '11',
        'diciembre': '12'
    }
    
    # Verificar si el mes ingresado es válido
    if mes not in meses_dict:
        return {'error': f"El mes '{mes}' no es válido"}
    
    # Obtener el formato numérico del mes
    mes_numerico = meses_dict[mes]

    # Filtrar el DataFrame por el mes de estreno
    peliculas_mes = df[df['release_date'].str.contains(f"-{mes_numerico}-")]

    # Obtener la cantidad de películas en el mes consultado
    cantidad_peliculas = len(peliculas_mes)

    # Devolver la cantidad de películas
    return {'mes': mes, 'cantidad': cantidad_peliculas}


@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia_semana:str):
    dias_en_espanol = {
        'lunes': 0,
        'martes': 1,
        'miercoles': 2,
        'jueves': 3,
        'viernes': 4,
        'sabado': 5,
        'domingo': 6
    }

    dia_semana = dia_semana.lower()
    if dia_semana not in dias_en_espanol:
        return "Día de la semana inválido"

    dia_semana_numero = dias_en_espanol[dia_semana]

    # Convertir la columna 'release_date' a tipo datetime si no lo es
    df['release_date'] = pd.to_datetime(df['release_date'])

    peliculas_dia = df[df['release_date'].dt.weekday == dia_semana_numero]
    cantidad = len(peliculas_dia)
    return {'dia':dia_semana.capitalize(), 'cantidad':cantidad}

@app.get('/score_titulo/{titulo}')
def score_titulo(titulo_de_la_filmacion):
    pelicula = df[df['title'] == titulo_de_la_filmacion].iloc[0]
    titulo = pelicula['title']
    año_estreno = pd.to_datetime(pelicula['release_date']).year
    score = pelicula['popularity']
    return {'titulo':titulo, 'anio':año_estreno, 'popularidad':score}    

@app.get('/votos_filmacion/{titulo}')
def votos_titulo(titulo_de_la_filmacion):
    pelicula = df[df['title'] == titulo_de_la_filmacion].iloc[0]
    titulo = pelicula['title']
    fecha_estreno = pd.to_datetime(pelicula['release_date'])
    año_estreno = fecha_estreno.year
    cantidad_votos = pelicula['vote_count']
    promedio_votos = pelicula['vote_average']
    if cantidad_votos >= 2000:
        return {'titulo':titulo, 'anio':año_estreno, 'voto_total':cantidad_votos, 'voto_promedio':promedio_votos}  
    else:
        return f"La película {titulo} no cumple con la condición de tener al menos 2000 valoraciones"
    
@app.get('/actor/{actor}')
def get_actor(nombre_actor):
    actor_films = df[df['actor_names'].notnull() & df['actor_names'].str.contains(nombre_actor, case=False)]['return']
    cantidad_filmaciones = actor_films.shape[0]
    retorno_total = actor_films.sum()
    promedio_retorno = actor_films.mean()
    return {'actor':nombre_actor, 'cantidad_filmaciones':cantidad_filmaciones, 'retorno_total':retorno_total, 'retorno_promedio':promedio_retorno}
    

@app.get('/director/{director}')
def get_director(nombre_director):
    dataset_filled = df.fillna('Unknown')  # Reemplazar los valores faltantes con 'Unknown'
    director_films = dataset_filled[dataset_filled['director'].str.contains(nombre_director, case=False)]
    cantidad_peliculas = director_films.shape[0]
    retorno_total = director_films['return'].sum()

    resultado = f"Director: {nombre_director}, Retorno Total: {retorno_total},"
    
    for _, row in director_films.iterrows():
        titulo = row['title']
        fecha_lanzamiento = row['release_date']
        retorno_individual = row['return']
        costo = row['budget']
        ganancia = row['revenue']
        
        resultado += f"Películas: {titulo} Año: {fecha_lanzamiento} Retorno individual: {retorno_individual} Costo: {costo} Ganancia: {ganancia}"
    
    return resultado






ml_df = ml_df.head(5000)

ml_df['genre_names'] = ml_df['genre_names'].fillna('')  # Reemplazar np.nan con una cadena vacía

title_vectorizer = TfidfVectorizer()
title_matrix = title_vectorizer.fit_transform(ml_df['title'])

genre_vectorizer = TfidfVectorizer()
genre_matrix = genre_vectorizer.fit_transform(ml_df['genre_names'])

combined_matrix = hstack([title_matrix, genre_matrix])

cosine_sim = cosine_similarity(combined_matrix, combined_matrix)


def get_recommendations(title, cosine_sim, df, top_n=5):
    index = ml_df[ml_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommendation_indices = [i for i, _ in sim_scores if i != index]
    recommendation_indices = recommendation_indices[:top_n]
    recommendations = df.iloc[recommendation_indices]['title']
    return recommendations


@app.get('/recomendacion/{recomendacion}')
def obtener_recomendacion(recomendacion: str):
    recommendations = get_recommendations(recomendacion, cosine_sim, ml_df)
    return {"recomendaciones": recommendations.tolist()}





