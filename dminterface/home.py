import streamlit as st

 

st.set_page_config(
    page_title='Acceuil',
    page_icon=(''),
    )




st.title('Bienvenue dans cette interface graphique (GUI)')
st.write("Elle est organisée en trois sections distinctes, chacune dédiée au traitement des ensembles de données dataset1, dataset2 et dataset3.")
st.write("Pour accéder aux onglets correspondants à chaque ensemble de données, veuillez vous diriger vers les pages situées à gauche.")


st.sidebar.success("Main page")

gif_url = "assets\exitedman.gif" 
st.image(gif_url)