import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

data = pd.read_csv('df_final_limpio.csv')

#2. Titulo de pagina
st.set_page_config(page_title="Sistema de recomendación de Recetas")

#3. Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        options=['Home', 'Visualizando los datos', 'Armado del modelo', 'Encontrá tu libro'],
    )

