import streamlit as st
from projetpython import *

# Set Streamlit page configuration
st.set_page_config(
    page_title='Dataset 2',
    page_icon='',
)

# Title and Sidebar
st.title('Analyse, prétraitement et visualisation des données temporelles')
st.sidebar.success("Dataset 2")

# Available attributes and visualizations
attributs = ["zcta", "time_period", "population", "Start date", "end date", "case count", "test count", "positive tests", "case rate", "test rate", "positivity rate"]
visualisations = ["Box Plot", "Histogramme", "Scatter Plot", "correlation"]

# Sidebar selection using selectbox
selected_attr = st.sidebar.selectbox("Sélectionnez un attribut", attributs)
selected_attr2 = st.sidebar.selectbox("Sélectionnez un deuxième attribut", attributs)
selected_visual = st.sidebar.selectbox("Sélectionnez un type de visualisation", visualisations)

# Load the dataset
dataset2 = load("Dataset2.csv")

rgb_color = (72, 120, 170)  # RGB values for teal
rgb_style = f"color: rgb{rgb_color};"
st.markdown(f"<h2 style='{rgb_style}'>Partie analyse</h2>", unsafe_allow_html=True)

# Analyze button
if st.button("Analyser"):
    resultats = analyse_dataset2(dataset2, selected_attr)
    st.write(f"**Résultats pour l'attribut {selected_attr} :**")
    if selected_attr != "Start date" and selected_attr != "end date":
        st.write(f"Q0 : {resultats[0]}, Q1 : {resultats[1]}, Q2 : {resultats[2]}, Q3 : {resultats[3]}, Q4 : {resultats[4]}")
        st.write(f"Moyenne : {resultats[7]}, Médiane : {resultats[8]}, Mode : {resultats[9]}")
        st.write(f"Nombre de valeurs manquantes : {resultats[5]}, Pourcentage de valeurs manquantes : {resultats[6]}%")
    else:
        st.write(f"Nombre de valeurs manquantes : {resultats[0]}, Pourcentage de valeurs manquantes : {resultats[1]}%")

# Visualize button
if st.button("Visualiser"):
    if selected_visual == "Scatter Plot":
        visualise_dataset2(selected_visual, dataset2, selected_attr, selected_attr2)
    else:
        visualise_dataset2(selected_visual, dataset2, selected_attr, selected_attr)

if st.button("Analyser apres preprocessing"):
    dataset2 = preprocessing2(dataset2)
    resultats = analyse_dataset2(dataset2, selected_attr)
    st.write(f"**Résultats pour l'attribut {selected_attr} :**")
    if selected_attr != "Start date" and selected_attr != "end date":
        st.write(f"Q0 : {resultats[0]}, Q1 : {resultats[1]}, Q2 : {resultats[2]}, Q3 : {resultats[3]}, Q4 : {resultats[4]}")
        st.write(f"Moyenne : {resultats[7]}, Médiane : {resultats[8]}, Mode : {resultats[9]}")
        st.write(f"Nombre de valeurs manquantes : {resultats[5]}, Pourcentage de valeurs manquantes : {resultats[6]}%")
    else:
        st.write(f"Nombre de valeurs manquantes : {resultats[0]}, Pourcentage de valeurs manquantes : {resultats[1]}%")

# Visualize button
if st.button("Visualiser apres preprocessing "):
    dataset2 = preprocessing2(dataset2)
    visualise_dataset2(selected_visual, dataset2, selected_attr, selected_attr2)


st.markdown(f"<h2 style='{rgb_style}'>Partie Visualisation </h2>", unsafe_allow_html=True)

if st.button("I- La distribution du nombre total des cas confirmés et tests positifs par zones"):
    dataset2=preprocessing2(dataset2)
    case_by_zone(dataset2)
texte_zone = st.text_input("Saisissez votre zone choisie", "95127")
if st.button("II- Evolution tests COVID-19, tests positifs,nombre de cas au fil du temps pour une zone choisit"):
        dataset2=preprocessing2(dataset2)
        zone_by_time(dataset2,int(texte_zone))

#Comment les cas covid positifs sont distribués par zone et par année
if st.button("III- Distribution des cas covid positifs par zone et année"):
        dataset2=preprocessing2(dataset2)
        visualisation_3(dataset2)

if st.button("IV- Rapport entre la population et le nombre de tests effectués"):
        dataset2=preprocessing2(dataset2)
        visualisation_4(dataset2)
    
if st.button("V- Les 5 zones les plus fortement impactées par le coronavirus"):
        dataset2=preprocessing2(dataset2)
        visualisation_5(dataset2)
texte_date1 = st.text_input("Saisissez la date de début",'2022-01-01')
texte_date2= st.text_input("Saisissez la date de fin",'2022-12-31')
if st.button("VI- Rapport entre les cas confirmés, les tests effectués et les tests positifs au fil du temps pour chaque zone"):
        dataset2=preprocessing2(dataset2)
        visualisation_6(dataset2,texte_date1,texte_date2)