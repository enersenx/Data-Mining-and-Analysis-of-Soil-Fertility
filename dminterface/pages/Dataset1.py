import streamlit as st
from projetpython import *


st.set_page_config(
    page_title='Dataset 1',
    page_icon=(''),
)
st.title('Application des méthodes d\'Apprentissage supervisé et non supervisé sur des données statiques ')
st.sidebar.success("Dataset 1")

# Load the dataset
dataset1 = load("Dataset1.csv")
eps_text="0"
min_samples_text="0"
k_value_text="0"
init_type_option="0"
distance_option="0"
num_trees_text="0"
# Available attributes and visualizations
attributs = ["N","P","K","pH","EC","OC","S","Zn","Fe","Cu","Mn","B","OM","Fertility"]
visualisations = ["Box Plot", "Histogramme", "Scatter Plot", "correlation"]

# Sidebar selection using selectbox
selected_attr = st.sidebar.selectbox("Sélectionnez un attribut", attributs)
selected_attr2 = st.sidebar.selectbox("Sélectionnez un deuxième attribut", attributs)
selected_visual = st.sidebar.selectbox("Sélectionnez un type de visualisation", visualisations)



rgb_color = (72, 120, 170)  # RGB values for teal
rgb_style = f"color: rgb{rgb_color};"
st.markdown(f"<h2 style='{rgb_style}'>Partie analyse et pretraitement</h2>", unsafe_allow_html=True)

# Analyze button
if st.button("Analyser"):
    resultats = analyse_dataset1(dataset1, selected_attr)
    st.write(f"**Résultats pour l'attribut {selected_attr} :**")
    st.write(f"Q0 : {resultats[0]}, Q1 : {resultats[1]}, Q2 : {resultats[2]}, Q3 : {resultats[3]}, Q4 : {resultats[4]}")
    st.write(f"Moyenne : {resultats[7]}, Médiane : {resultats[8]}, Mode : {resultats[9]}")
    st.write(f"Nombre de valeurs manquantes : {resultats[5]}, Pourcentage de valeurs manquantes : {resultats[6]}%")
   
# Visualize button
if st.button("Visualiser"):
    if selected_visual == "Scatter Plot":
        visualise_dataset2(selected_visual, dataset1, selected_attr, selected_attr2)
    else:
        visualise_dataset2(selected_visual, dataset1, selected_attr, selected_attr)


treat_outliers_option = st.selectbox("Choisissez la méthode de traiter les outliers", ["binning", "winsorizing"])
normalize_option = st.selectbox("Choisissez la méthode de normalisation", ["z_score_normalization", "min_max_normalize"])

if st.button("Analyser apres preprocessing"):
    
    if normalize_option=="z_score_normalization":
        #preprocessing1(dataset1,'binning',z_score_normalization)
        dataset1=preprocessing1(dataset1,treat_outliers_option,z_score_normalization)
        resultats = analyse_dataset2(dataset1, selected_attr)
        st.write(f"**Résultats pour l'attribut {selected_attr} :**")
        st.write(f"Q0 : {resultats[0]}, Q1 : {resultats[1]}, Q2 : {resultats[2]}, Q3 : {resultats[3]}, Q4 : {resultats[4]}")
        st.write(f"Moyenne : {resultats[7]}, Médiane : {resultats[8]}, Mode : {resultats[9]}")
        st.write(f"Nombre de valeurs manquantes : {resultats[5]}, Pourcentage de valeurs manquantes : {resultats[6]}%")
    elif normalize_option=="min_max_normalize":
        #preprocessing1(dataset1,'binning',z_score_normalization)
        dataset1=preprocessing1(dataset1,treat_outliers_option,min_max_normalize)
        resultats = analyse_dataset2(dataset1, selected_attr)
        st.write(f"**Résultats pour l'attribut {selected_attr} :**")
        st.write(f"Q0 : {resultats[0]}, Q1 : {resultats[1]}, Q2 : {resultats[2]}, Q3 : {resultats[3]}, Q4 : {resultats[4]}")
        st.write(f"Moyenne : {resultats[7]}, Médiane : {resultats[8]}, Mode : {resultats[9]}")
        st.write(f"Nombre de valeurs manquantes : {resultats[5]}, Pourcentage de valeurs manquantes : {resultats[6]}%")
 

#Visualize button
if st.button("Visualiser apres preprocessing "):
    if normalize_option=="z_score_normalization":
        dataset1=preprocessing1(dataset1,treat_outliers_option,z_score_normalization)
        visualise_dataset2(selected_visual, dataset1, selected_attr, selected_attr2)
    elif normalize_option=="min_max_normalize":
        dataset1=preprocessing1(dataset1,treat_outliers_option,min_max_normalize)
        visualise_dataset2(selected_visual, dataset1, selected_attr, selected_attr2)







def plot_sl(dataset,y_pred):
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)

    st.title('Clustering Predictions')

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis')

    plt.title(' Clustering Predictions')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    st.pyplot(fig)

def plot_sl_kmeans(dataset,y_pred):
  
    X = dataset.copy()

    X = X.drop('Fertility', axis=1)

    # Apply PCA for dimensionality reduction (adjust n_components as needed)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Streamlit app
    st.title('Clustering Predictions')

    # Scatter plot in the reduced 2D space
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis')

    # Plot customization
    plt.title('Clustering Predictions')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Display the plot using Streamlit
    st.pyplot(fig)

def plot_sldbscan(dataset,y_pred):
  
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X = dataset.copy()
    X = X.drop('Fertility', axis=1)
    X_pca = pca.fit_transform(X)


    # Streamlit app
   # Convert the reduced dataset to a DataFrame
    df = pd.DataFrame(X_pca, columns=['Principal Component 1', 'Principal Component 2'])
    df['Cluster'] = y_pred

    # Streamlit app
    st.title('DBSCAN Clustering with PCA - Discretized Dataset')

    # Visualize the clusters
    fig, ax = plt.subplots()
    scatter = sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Cluster', data=df, palette='viridis', legend='full', ax=ax)

    # Plot customization
    plt.title('DBSCAN Clustering with PCA - Discretized Dataset')

    # Display the plot using Streamlit
    st.pyplot(fig)

def stknn(method,dataset1,eps,min_samples,k_value,init_type,distance_type,num_trees):
    if normalize_option=="z_score_normalization":
        dataset1=preprocessing1(dataset1,treat_outliers_option,z_score_normalization)
        #y_pred_knn = execute_knn(dataset1,distance_option,int(k_value_text))
        epsfloat=float(eps)

        y_pred=choose_classification(method,dataset1,epsfloat,int(min_samples),int(k_value),init_type,distance_type,int(num_trees))
        if method=="knn" or method=="decision_tree" or method=="random_forest":
            plot_sl(dataset1,y_pred)
            dfknn=pd.DataFrame(y_pred)
            knn_results_stacked = dfknn.stack()
            confusionmatrix_knn=valeurs_matrice_confusion(dataset1,knn_results_stacked)
            plot_conf_mat(confusionmatrix_knn)
            measures_classwise_1, measures_global_1 = mesures_evaluation(confusionmatrix_knn)
            st.write(f"Mesures accuracy pour chaque classe")
            for class_name, metrics in measures_classwise_1.items():
                st.write(f"{class_name}: {metrics['Accuracy']:.2%}")
            st.write(f"mesures globales 1",measures_global_1)
        elif method=="kmeans":
            plot_sl_kmeans(dataset1,y_pred)
            df=pd.DataFrame(y_pred)
            results_stacked = df.stack()
            silhouette_score = silhouette_coefficient(dataset1,results_stacked )
            st.write(f"mesures silhouette",silhouette_score)
        else:
            plot_sldbscan(dataset1,y_pred)
            result_labels=execute_dbscan(dataset1,float(eps),int(min_samples))
            silhouette_score = silhouette_coefficient(dataset1,result_labels )
            st.write(f"mesures silhouette",silhouette_score)
        

    elif normalize_option=="min_max_normalize":
        dataset1=preprocessing1(dataset1,treat_outliers_option,min_max_normalize)
        #y_pred_knn = execute_knn(dataset1,distance_option,int(k_value_text))
        epsfloat=float(eps)

        y_pred=choose_classification(method,dataset1,epsfloat,int(min_samples),int(k_value),init_type,distance_type,int(num_trees))
        if method=="knn" or method=="decision_tree" or method=="random_forest":
            plot_sl(dataset1,y_pred)
            dfknn=pd.DataFrame(y_pred)
            knn_results_stacked = dfknn.stack()
            confusionmatrix_knn=valeurs_matrice_confusion(dataset1,knn_results_stacked)
            plot_conf_mat(confusionmatrix_knn)
            measures_classwise_1, measures_global_1 = mesures_evaluation(confusionmatrix_knn)
            st.write(f"Mesures accuracy pour chaque classe")
            for class_name, metrics in measures_classwise_1.items():
                st.write(f"{class_name}: {metrics['Accuracy']:.2%}")
            st.write(f"mesures globales 1",measures_global_1)
        elif method=="kmeans":
            plot_sl_kmeans(dataset1,y_pred)
            df=pd.DataFrame(y_pred)
            results_stacked = df.stack()
            silhouette_score = silhouette_coefficient(dataset1,results_stacked )
            st.write(f"mesures silhouette",silhouette_score)
        else:
            plot_sldbscan(dataset1,y_pred)
            result_labels=execute_dbscan(dataset1,float(eps),int(min_samples))
            silhouette_score = silhouette_coefficient(dataset1,result_labels )
            st.write(f"mesures silhouette",silhouette_score)







st.markdown(f"<h2 style='{rgb_style}'>Partie Classification</h2>", unsafe_allow_html=True)

rgb_green_color=(63,128,70)
rgb_green_color_clair=(133,196,120)
rgb_green_color_clair_style=f"color: rgb{rgb_green_color_clair};"
rgb_green_color_style=f"color: rgb{rgb_green_color};"

st.markdown(f"<h3 style='{rgb_green_color_style}'> Analyse supervisée</h3>", unsafe_allow_html=True)
st.markdown(f"<h4 style='{rgb_green_color_clair_style}'> knn</h4>", unsafe_allow_html=True)
distance_option = st.selectbox("Distance", ['manhattan','euclidean','minkowski','cosine','hamming'])
k_value_text = st.text_input("valeur k", 5)
if st.button("Executer Knn ") :
    st.write(f"Patience...")
    stknn("knn",dataset1,eps_text,min_samples_text,k_value_text,init_type_option,distance_option,num_trees_text)


st.markdown(f"<h4 style='{rgb_green_color_clair_style}'> Decision tree </h4>", unsafe_allow_html=True)


if st.button("Executez Decision Tree"):
    st.write(f"Patience...")
    stknn("decision_tree",dataset1,eps_text,min_samples_text,k_value_text,init_type_option,distance_option,num_trees_text)

st.markdown(f"<h4 style='{rgb_green_color_clair_style}'> Random Forest</h4>", unsafe_allow_html=True)
num_trees_text= st.text_input("num trees", 5)
if st.button("Executez Random Forest"):
    st.write(f"Patience...")
    stknn("random_forest",dataset1,eps_text,min_samples_text,k_value_text,init_type_option,distance_option,num_trees_text)

st.markdown(f"<h3 style='{rgb_green_color_style}'> Analyse non supervisée</h3>", unsafe_allow_html=True)
st.markdown(f"<h4 style='{rgb_green_color_clair_style}'>kmeans</h4>", unsafe_allow_html=True)
init_type = st.selectbox("type", ["kmeans_plusplus_init","init_centroids"])
if st.button("Executez kmeans"): 
    st.write(f"Patience...")
    if init_type=="kmeans_plusplus_init":
        stknn("kmeans",dataset1,eps_text,min_samples_text,k_value_text,kmeans_plusplus_init,distance_option,num_trees_text)
    else:
        stknn("kmeans",dataset1,eps_text,min_samples_text,k_value_text,init_centroids,distance_option,num_trees_text)
    

st.markdown(f"<h4 style='{rgb_green_color_clair_style}'>DBSCAN</h4>", unsafe_allow_html=True)
eps_text = st.text_input("valeur eps", 0.2)
min_samples_text = st.text_input("min samples", 5)
if st.button("DBSCAN"):
    st.write(f"Patience...")
    stknn("dbscan",dataset1,eps_text,min_samples_text,k_value_text,init_type_option,distance_option,num_trees_text)



def choose_classification(method,dataset,eps,min_samples,k_value,init_type,distance_type,num_trees):
    if method=="dbscan":
        return execute_dbscan(dataset,eps,min_samples)
    elif method=="kmeans":
        return execute_kmeans(dataset,k_value,distance_type,init_type)
    elif method=="knn":
        return execute_knn(dataset,distance_type,k_value)
    elif method=="decision_tree":
        return execute_decision_tree(dataset)
    elif method=="random_forest":
        return execute_random_forest(dataset,num_trees)


