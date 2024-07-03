import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.colors as colors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk


data = pd.read_csv('df_final_limpio.csv')


#2. Titulo de pagina
st.set_page_config(page_title="Sistema de recomendación de Recetas")

#3. Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        options=['Introducción', 'Visualización', 'Modelo', 'Encontrá tu proxima receta'],
    )

### agregue imagen prueba
import requests
from PIL import Image
from io import BytesIO
import streamlit as st


# URL de la imagen del experimento de doble rendija
image_url = "https://humanidades.com/wp-content/uploads/2018/12/veganismo-1-e1585011571609.jpg"
image_url2= "https://thermomix-majadahonda.es/media/Posts/attachments/ce90da2128f36c2d5d2838b76a3c8942.jpg"
# Cargar la imagen desde la URL
response = requests.get(image_url)
response2= requests.get(image_url2)


image = Image.open(BytesIO(response.content))
image2 = Image.open(BytesIO(response2.content))
####################################################################
if selected == 'Introducción':
    st.title('Storytelling')
    st.write('Como 2 amigos emprendedores 😊 , siempre nos hacemos preguntas para ver que podemos solucionar. Un día, nos encontrabamos en una cocina y salió la pregunta') 
    st.subheader('¿que quieren cenar hoy?')

    st.write('Esta fue la pregunta que se nos llevo  a crear este sistema de recomendacion, y vimos que en nuestras casas a diario nos preguntabamos que podíamos cenar hoy con los ingredientes que tenemos.')
    pregunta = st.image("pregunta.jpeg")
    st.write('Pensamos en como podiamos solucionar este problema... nos pusimos manos a la obra y con lo que aprendimos en IA apareció la brillante idea.')
    st.write('.............................................................................................................................................................................')
    st.header('🍔🍕🌭 Sistema de recomendacion de recetas 🍲🥘🌯')
    st.write('Ante la cantidad de recetas que tenemos en nuestra querido internet, es ideal que podamos mostrarte cuales son las recetas que podes hacer con tus ingredientes y en base a eso elegir cual preferis.')
    st.write('Por eso, buscamos desarrollar un sistema que te ayude a elegir de la mejor manera.')
    st.image(image2)

    st.header('Explicando los datos 🗣️📊:')
    st.write('El conjunto de datos utilizado, es un conjunto de datos sacado de kaggle (https://www.kaggle.com/datasets/kanaryayi/recipe-ingredients-and-reviews?resource=download).')
    st.write('Contiene recetas, con sus ingredientes y direcciones, y despues los ratings')
    st.write('Para este proyecto se utiliza una base de 12.351 recetas y 291.840 puntuaciones que realizaron usuarios acerca de los mismos.')
    st.write("A continuación podemos ver cómo se componen los set de datos utilizados:")
    st.dataframe(data.head())


    st.subheader("\n Descripción de las columnas.")
    st.markdown("\n**DataFrame final**")
    st.markdown("\n **Recipe ID**:  Número de identificación de la receta.")
    st.markdown("\n **Recipe Name** : Nombre de la receta.")
    st.markdown("\n **Ingredientes** : ingredientes que tiene cada receta ")
    st.markdown("\n **Directions** : Pasos a seguir para hacer la receta.")
    st.markdown("\n **Rate** : Puntuación de cada receta promedio.")

    st.header('¿Vegana o no vegana?')
    st.write('Otra de las dudas que nos surgian, es si teniamos invitados veganos en casa poder hacerles una receta adecuada para ellos.')    
    st.image(image, caption="Figura 1. El veganismo es una filosofía de vida que busca excluir la explotación animal para alimentación, vestimenta, entretenimiento o cualquier otro propósito, abogando por una dieta basada en vegetales y un estilo de vida compasivo y respetuoso con todos los seres sintientes.", use_column_width=True)
###########################################################################

elif selected == 'Visualización':
    st.title('Visualizando los datos')
    
    puntuacion = st.image("puntuacion.png")
    st.markdown("Vemos que la mayoria de los datos son valiosos ya que la mayoria de los ratings fueron positivos.")
    st.markdown("Por lo tanto para el modelo de recomendacion decididimos quedarnos solo con las recetas de rating mayor a 3.")

    st.header("Hicimos nube palabras para ver los ingredientes mas utilizados en las recetas segun los distintos ratings:")
    
    st.write('Nube de palabras para las recetas con rating 3:')
    cloud3 = st.image("cloud_3.png")
    st.markdown("Vemos que sal, huevo y leche son los ingredientes mas utilizados en las recetas con rating 3.")
    st.write('Nube de palabras para las recetas con rating 4:')
    cloud4 = st.image("cloud_4.png")
    st.markdown("Vemos que agua, azucar blanca y pimienta negra son los ingredientes mas utilizados en las recetas con rating 4.")
    st.write('Nube de palabras para las recetas con rating 5:')
    cloud5 = st.image("cloud_5.png")
    st.markdown("Vemos que las utilizados en las recetas con rating 5 son iguales que las recetas que tienen rating 4.")

    st.write('Tiempo de coccion de las recetas')
    tiempo_cocc = st.image("tiempo_coccion.jpeg")
    st.markdown("Vemos que la mayoria de las recetas tienen tiempo de coccion de menos de 3hs ")

    st.write('Cantidad de rates que pusieron las personas')
    tiempo_cocc = st.image("cantidad_de_rates.png")
    st.markdown("Vemos que la mayoria de las personas rateo menos de 5 recetas, lo que resulta dificil hacer un modelo que devuelva recetas parecidas a los gustos de ellos ya que no demuestran mucho lo que les gusta reteando tan pocas recetas.")
    
    st.write('Recetas con mas rating')
    tiempo_cocc = st.image("recetas_masrate.jpeg")
    st.markdown("Vemos que hay una receta que se llama restaurant y tiene 7 personas que la ratearon, lo que resulta raro pero estaba en la lista de palabras dentro de los nombres de recetas")
    
#####################################################
elif selected == 'Modelo':
   # Título de la aplicación
    st.write('https://colab.research.google.com/drive/1cPJhgMGfMNvOcDgmATnn239pLh3zdLdW#scrollTo=TSjRQkT449F3')
    st.title("Ventajas y Desventajas de Usar Solo la Columna Ingredients vs. Ingredients + Directions")

# Ventajas y Desventajas de Usar Solo la Columna Ingredients
    st.header("Usar Solo la Columna Ingredients")
    st.subheader("Ventajas:")
    st.markdown("""
    - *Simplicidad*: El modelo es más simple y más rápido de entrenar y evaluar.
    - *Relevancia*: Los ingredientes son directamente relevantes para los usuarios que buscan recetas basadas en lo que tienen disponible.
    - *Menor Dimensionalidad*: Menos datos que procesar, lo que puede mejorar la eficiencia.
    """)

    st.subheader("Desventajas:")
    st.markdown("""
    - *Menos Información*: Se pierde el contexto y las instrucciones de cómo preparar la receta.
    - *Menor Precisión*: Puede haber muchas recetas con los mismos ingredientes pero diferentes métodos de preparación, lo que puede afectar la precisión de la recomendación.
    """)

    # Ventajas y Desventajas de Usar Ingredients + Directions
    st.header("Usar Ingredients + Directions")
    st.subheader("Ventajas:")
    st.markdown("""
    - *Mayor Contexto*: Las direcciones proporcionan más contexto sobre cómo preparar la receta, lo que puede ayudar a diferenciar recetas similares.
    - *Mejor Relevancia*: Puede ayudar a recomendar recetas que no solo tienen los ingredientes correctos sino que también se ajustan al método de preparación preferido por el usuario.
    - *Más Completo*: El modelo tiene más información para aprender y hacer recomendaciones más precisas.
    """)

    st.subheader("Desventajas:")
    st.markdown("""
    - *Mayor Complejidad*: El modelo es más complejo y puede ser más lento de entrenar y evaluar.
    - *Dimensionalidad Alta*: Más datos que procesar, lo que puede requerir más recursos computacionales.
    - *Ruido*: Las direcciones pueden contener más "ruido" o información irrelevante para la recomendación basada en ingredientes.
    """)




######################################################    

elif selected == 'Encontrá tu proxima receta':
   
   # Asegurarse de que Ingredients esté como listas de strings si no lo está ya
    data['Ingredients'] = data['Ingredients'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')])

# Manejar los valores nulos en la columna Directions
    data['Directions'] = data['Directions'].fillna('')  # Rellenar valores nulos con cadena vacía

# Convertir Directions a listas de strings separadas por puntos
    data['Directions'] = data['Directions'].apply(lambda x: [step.strip() for step in x.split('.') if step.strip()])  # Separar por puntos y limpiar espacios

# Concatenar ingredientes y direcciones en un solo campo para la vectorización
    data['Ingredients_and_Directions'] = data.apply(
    lambda row: ' '.join(map(str, row['Ingredients'])) + ' ' + ' '.join(row['Directions']), axis=1)

# Vectorización de los ingredientes y direcciones
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['Ingredients_and_Directions'])

# Lista de ingredientes no veganos comunes
    non_vegan_ingredients = ['cheese', 'milk', 'egg', 'butter', 'cream', 'meat', 'chicken', 'beef', 'pork', 'fish', 'honey']

# Función para recomendar recetas basadas en ingredientes ingresados por el usuario y verificar veganeidad
    def recommend_recipes(user_ingredients):
    # Transformar los ingredientes del usuario utilizando el mismo TF-IDF
        user_tfidf = tfidf.transform([' '.join(user_ingredients)])

    # Calcular similitud de coseno entre el vector del usuario y todos los vectores de recetas
        cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Obtener los índices de las recetas ordenadas por similitud (mayor a menor)
        similar_indices = cosine_similarities.argsort()[::-1]

    # Recolectar las recetas recomendadas y sus similitudes
        recommended_recipes = []
        for i in similar_indices:
            recipe_name = data.iloc[i]['Recipe Name']
            cosine_sim = cosine_similarities[i]
            ingredients = data.iloc[i]['Ingredients']
            rate = data.iloc[i]['Rate']

        # Calcular ingredientes faltantes
            missing_ingredients = [ingredient for ingredient in ingredients if ingredient.strip().lower() not in [ing.lower() for ing in user_ingredients]]

        # Verificar si la receta es vegana
            is_vegan = all(ingredient.lower() not in non_vegan_ingredients for ingredient in ingredients)

        # Incluir solo recetas con un Rate mayor o igual a 3
            if rate >= 3:
                recommended_recipes.append((recipe_name, cosine_sim, is_vegan, rate, missing_ingredients))

        return recommended_recipes[:5]  # Devolver solo las 5 primeras recetas

# Configurar la aplicación de Streamlit
    st.title('Recomendador de Recetas')
    st.write('Ingresa tus ingredientes y te recomendaremos recetas!')

# Ingreso de ingredientes por parte del usuario
    user_input = st.text_input("Ingresa los ingredientes separados por comas:")
    user_ingredients = [ingredient.strip() for ingredient in user_input.split(',')]

# Ejecutar el proceso de recomendación basado en los ingredientes ingresados por el usuario
    if st.button('Recomendar'):
        recommended_recipes = recommend_recipes(user_ingredients)

    # Mostrar las recetas recomendadas, sus similitudes, si son veganas, su rate y los ingredientes faltantes
        st.write("\nRecetas recomendadas:")
        for recipe, similarity, vegan, rate, missing_ingredients in recommended_recipes:
            vegan_status = "Vegana" if vegan else "No Vegana"
            st.write(f'Receta: {recipe} | Similitud: {similarity:.2f} | Vegana: {vegan_status} | Rate: {rate}')
            st.write(f'Ingredientes que te faltan: {", ".join(missing_ingredients)}')