import streamlit as st
from projetpython import *
 

st.set_page_config(
    page_title='Dataset 3 ',
    page_icon=(''),
    )

st.title('Extraction de motifs fréquents, règles d’associations et corrélations') 
st.sidebar.success("Dataset 3")
#Discretiser un attribut au choix: ‘Temperature’ ou bien ‘Rainfall’ ou bien 
#‘Humidity’, puis discrétiser ces données continues de cet attribut, en utilisant ces 2 méthodes : 
#- En classes d’effectifs égaux (equal-frequency).
#- En classes d’amplitudes égales (equal-width).
# Available attributes and visualizations
attributs = ["Temperature","Rainfall","Humidity"]



# Load the dataset
dataset3 = load("Dataset2.csv")

rgb_color = (72, 120, 170)  # RGB values for teal
rgb_style = f"color: rgb{rgb_color};"
st.markdown(f"<h2 style='{rgb_style}'>Partie analyse</h2>", unsafe_allow_html=True)

st.write(f"Dataset avant preprocessing")
dataset3=load_excel("Dataset3.xlsx")
st.table(dataset3.head(5))
    

discretise_option = st.selectbox("Choisissez la méthode de discrétisation", ["equalfrequency", "equalwidth"])
datasetp=preprocessing3(dataset3,discretise_option)
st.write("dataset discretisé avec la méthode ",discretise_option)
st.table(datasetp.head(5))

st.markdown(f"<h2 style='{rgb_style}'>Extraction des motifs fréquents</h2>", unsafe_allow_html=True)

conf_option=st.selectbox("Choisissez la méthode de calcul de la confiance",["calculate_confidence","calculate_all_confidence","calculate_max_confidence","calculate_cosine_similarity","calculate_jaccard_similarity","calculate_kulczynski_similarity_union","calculate_lift"])
min_support_text = st.text_input("Saisissez le support min", "0.1")
min_conf_text = st.text_input("Saisissez la confiance min", "0.6")
tabs = st.container()


if st.button("Retourner les frequent itemsets"):
    min_support=float(min_support_text)
    all_frequent_itemsets, dataset3 = run_apriori(dataset3,min_support,discretise_option)
    st.write(f"Les itemsets frequents:",all_frequent_itemsets)

st.markdown(f"<h2 style='{rgb_style}'>Extraction des règles d’association et des fortes règles d’association</h2>", unsafe_allow_html=True)


if st.button("Retourner regles d'associations et les fortes regles :"):
    min_support=float(min_support_text)
    min_conf=float(min_conf_text)
    if conf_option=="calculate_confidence":
        dataset, fortes_regles,association_rules=run_all(dataset3,discretise_option,calculate_confidence,min_support,min_conf)
    elif conf_option=="calculate_all_confidence":
        dataset, fortes_regles,association_rules=run_all(dataset3,discretise_option,calculate_all_confidence,min_support,min_conf)
    elif conf_option=="calculate_max_confidence":
        dataset, fortes_regles,association_rules=run_all(dataset3,discretise_option,calculate_max_confidence,min_support,min_conf)
    elif conf_option=="calculate_cosine_similarity":
        dataset, fortes_regles,association_rules=run_all(dataset3,discretise_option,calculate_cosine_similarity,min_support,min_conf)
    elif conf_option=="calculate_jaccard_similarity":
        dataset, fortes_regles,association_rules=run_all(dataset3,discretise_option,calculate_jaccard_similarity,min_support,min_conf)
    elif conf_option=="calculate_kulczynski_similarity_union":
        dataset, fortes_regles,association_rules=run_all(dataset3,discretise_option,calculate_kulczynski_similarity_union,min_support,min_conf)
    elif conf_option=="calculate_lift":
        dataset, fortes_regles,association_rules=run_all(dataset3,discretise_option,calculate_lift,min_support,min_conf)
    st.write(f"Les regles d'association:",association_rules)
    st.write(f"Les fortes regles d'assiciation:",fortes_regles)
    
      
st.markdown(f"<h2 style='{rgb_style}'>Insertion de nouvelles observations</h2>", unsafe_allow_html=True)

st.write(f"Veillez inserer une nouvelle observation")
temp_text = st.text_input("Saisissez le support min", 25.5)
hum_text = st.text_input("Saisissez le support min", 60.0)
rain_text = st.text_input("Saisissez le support min", 1.2)
soil_text = st.text_input("Saisissez le support min", "Loam")
crop_text = st.text_input("Saisissez le support min", "Coconut")
fertilizer_text = st.text_input("Saisissez le support min", "DAP")


if st.button("En deduire une recommandation:"):
    min_support=float(min_support_text)
    if conf_option=="calculate_confidence":
        new_dataset,recommandations = recommendation(min_support,dataset3,discretise_option,calculate_confidence,temp_text,hum_text,rain_text,soil_text,crop_text,fertilizer_text)
    elif conf_option=="calculate_all_confidence":
        new_dataset,recommandations = recommendation(min_support,dataset3,discretise_option,calculate_all_confidence,temp_text,hum_text,rain_text,soil_text,crop_text,fertilizer_text)
    elif conf_option=="calculate_max_confidence":
        new_dataset,recommandations = recommendation(min_support,dataset3,discretise_option,calculate_max_confidence,temp_text,hum_text,rain_text,soil_text,crop_text,fertilizer_text)
    elif conf_option=="calculate_cosine_similarity":
        new_dataset,recommandations = recommendation(min_support,dataset3,discretise_option,calculate_cosine_similarity,temp_text,hum_text,rain_text,soil_text,crop_text,fertilizer_text)
    elif conf_option=="calculate_jaccard_similarity":
        new_dataset,recommandations = recommendation(min_support,dataset3,discretise_option,calculate_jaccard_similarity,temp_text,hum_text,rain_text,soil_text,crop_text,fertilizer_text)
    elif conf_option=="calculate_kulczynski_similarity_union":
        new_dataset,recommandations = recommendation(min_support,dataset3,discretise_option,calculate_kulczynski_similarity_union,temp_text,hum_text,rain_text,soil_text,crop_text,fertilizer_text)
    elif conf_option=="calculate_lift":
        new_dataset,recommandations = recommendation(min_support,dataset3,discretise_option,calculate_lift,temp_text,hum_text,rain_text,soil_text,crop_text,fertilizer_text)
    st.table(new_dataset.head(5))
    st.write(f"Top reccomandations",recommandations)
