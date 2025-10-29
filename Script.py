# -*- coding: utf-8 -*-
"""
@author: Alejandra Barrios
date: 17-10-2025

"""

import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import os 

#--- Configuración de la página ---
st.set_page_config(page_title="Clustering de Sensores", layout="wide")

#--- Definiciones de rutas --- 
model_path = "kmeans_model.pkl"
csv_path = "synthetic_data.csv"
scaler_path = "scaler.pkl"
datos_originales = "Mine_Dataset_2.xlsx"
data_cluster = "datos_cluster.csv"
#output_dir_data = "/content/"

#output_dir_mo = '/content/drive/MyDrive/Magister/Taller de aplicaciones/Modelo/'
#model_path = os.path.join(output_dir_mo, "kmeans_model.pkl")
#scaler_path = os.path.join(output_dir_mo, 'scaler.pkl')
#output_dir_data = '/content/drive/MyDrive/Magister/Taller de aplicaciones/'
#csv_path = os.path.join(output_dir_data, 'synthetic_data.csv')

# CREAR DOS COLUMNAS PRINCIPALES---
col1, col2 =st.columns([1,1.2]) # un poco mas ancha la derecha.

# =============================================
# COLUMNA IZQUIERDA: ESTADISTICAS DESCRIPTIVAS 
# =============================================

with col1:
    st.markdown ("## 1. Descripción de los datos")
    st.markdown ("""  El conjunto de datos **Land Mine** proviene del repositorio [UCI Machine learning Repository]
                 (https://archive.ics.uci.edu/dataset/763/land+mines-1) y corresponde a un problema de **clasificación** aplicado al área de **defensa militar**.
                 Contiene **338 instancias** con cuatro variables, tres de entrada -**Voltage (V)**, **High (H)** y **Soil Type (S)**- 
                 y una de salida —**Mine Type (M)**—, que representa el tipo de mina detectada. Los valores de voltaje corresponden a lecturas del sensor FLC ante distorsiones 
                 magnéticas, mientras que la altura se mide desde el suelo y el tipo de suelo se clasifica según su nivel de humedad. 
                 El objetivo es **predecir el tipo de mina** (por ejemplo, Anti-tank, Anti-personnel o sin mina) a partir de las condiciones del entorno y las mediciones del sensor, facilitando 
                 el desarrollo de sistemas inteligentes de detección automática de minas terrestres""")
                 
    st.markdown ("""### 1.1 Atributos de los datos""")

    st.markdown (""" **Voltage (V):**  
                 - Rango original: [0 V, 10.6 V]  
                 - Representa la salida del sensor FLC que mide la distorsión magnética.  
                 - En el dataset, el voltaje está **normalizado entre 0 y 1**.""")
                
    st.markdown (""" **High (H):**  
                - Rango original: [0 cm, 20 cm]  
                - Corresponde a la **altura del sensor** respecto al suelo.  
                - También se encuentra **normalizada entre 0 y 1**.""")

    st.markdown (""" **Soil Type (S):**  
                - Valores discretos de **1 a 6**, según humedad y composición:  
                    1 = Dry & Sandy  
                    2 = Dry & Humus  
                    3 = Dry & Limy  
                    4 = Humid & Sandy  
                    5 = Humid & Humus  
                    6 = Humid & Limy  
                - En los datos normalizados, estos valores se transforman a un rango de **0 a 1 en pasos de 0.2**.""")

    st.markdown (""" **Mine Type (M):**  
                - Variable dependiente (target), con **5 clases** que representan distintos tipos de minas.  
                - Codificada del **1 al 5**.""")
                 
                 
    st.markdown ("## 2. Estadísticas descriptivas de los datos originales")
    
    if os.path.exists(csv_path):
        df = pd.read_excel(datos_originales)

        st.write("### 2.1 Vista previa:")
        st.dataframe(df.head())

        st.write("### 2.2 Resumen estadístico:")
        st.dataframe(df.describe().T)

        st.write("### 2.3 Distribución por tipo de mina (M):")
        st.bar_chart(df['M'].value_counts().sort_index())
    
        # -------------------------------
        # 6. Mapa de correlaciones entre variables
        # -------------------------------
        st.write("### 2.4 Mapa de correlaciones entre variables")
        num_col = df.select_dtypes(include=np.number).columns.tolist()
        plt.figure(figsize=(12, 6))
        sns.heatmap(df[num_col].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
        plt.title("Correlación entre variables")
        plt.tight_layout()
        st.pyplot(plt.gcf())  # Renderiza el gráfico en Streamlit
        plt.clf()  # Limpiar figura para no sobreponer gráficos

        # -------------------------------
        # 7. Pairplot: relaciones entre variables y tipos de minas
        # -------------------------------
        st.write("### 2.5 Relaciones entre variables y clases (Pairplot)")
        pairplot_fig = sns.pairplot(df, hue="M", diag_kind="kde", plot_kws={"alpha":0.5})
        pairplot_fig.fig.suptitle("Relaciones entre variables y clases", y=1.02)
        st.pyplot(pairplot_fig)  # Renderiza el pairplot en Streamlit
        plt.close()  # Cierra la figura para liberar memoria
    
        # =========================================================
        # Apartado: Visualización de los clusters originales
        # =========================================================
        st.markdown("## 3. Resultados promedio de los clusters entrenados")
        
        # Datos de ejemplo según estadísticas de los clusters originales
        cluster_means = pd.DataFrame({
            'Cluster': [0, 1, 2, 3, 4],
            'V': [0.403539, 0.360214, 0.389561, 0.336420, 0.896815],
            'H': [0.241047, 0.775974, 0.251018, 0.773836, 0.272727],
            'S': [0.800000, 0.797619, 0.185075, 0.195122, 0.564103]
            })
        
        # Crear scatterplot H vs S con tamaño según V y color por cluster
        plt.figure(figsize=(8,6))
        sns.scatterplot(
            data=cluster_means,
            x='H', y='S',
            size='V', sizes=(100, 1000),
            hue='Cluster', palette='Set2', legend='full',
            marker='X'
            )

        # Etiquetas de cada cluster
        for i, row in cluster_means.iterrows():
            plt.text(row['H']+0.01, row['S']+0.01, f"C{row['Cluster']}", fontsize=10)

            plt.title("Clusters: Relación H vs S (tamaño según V)")
            plt.xlabel("H (Promedio por Cluster)")
            plt.ylabel("S (Promedio por Cluster)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)

        # Mostrar gráfico en Streamlit
        st.pyplot(plt)
    
        # =========================================================
        # Descripción del gráfico
        # =========================================================
        st.markdown("""
                    **En resumen:**
    
        - **Clusters 0 y 1:** ambos tienen `S` alto (~0.8), pero se diferencian porque el **Cluster 0** tiene `H` bajo y el **Cluster 1** tiene `H` alto.  
        - **Clusters 2 y 3:** ambos con `S` bajo (~0.2), pero se diferencian porque el **Cluster 2** tiene `H` bajo y el **Cluster 3** tiene `H` alto.  
        - **Cluster 4:** es único porque tiene `V` muy alto (~0.9), mientras que en los otros clusters `V` ronda entre 0.33 y 0.40.
        """)
        
        data = pd.read_csv(data_cluster)
        st.markdown(" ### 3.1 Caracterizando cada Cluster")

        # Calcular estadísticas por cluster
        cluster_stats = data.groupby('Cluster').agg({
            'V': ['mean', 'median', 'std', 'min', 'max'],
            'H': ['mean', 'median', 'std', 'min', 'max'],
            'S': ['mean', 'median', 'std', 'min', 'max']
            })

        # Mostrar las estadísticas en Streamlit
        st.write("### 3.2 Estadísticas descriptivas por Cluster")
        st.dataframe(cluster_stats.style.format("{:.2f}"))

        # Graficar la distribución de cada variable por cluster
        for col in ['V', 'H', 'S']:
            st.markdown(f"#### Distribución de **{col}** por Cluster")
    
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='Cluster', y=col, data=data, palette='Set2', ax=ax)
    
            # Calcular y superponer las medias
            means = data.groupby('Cluster')[col].mean().reset_index()
            sns.scatterplot(x='Cluster', y=col, data=means,
                    color='red', marker='o', s=100, zorder=10, label='Media', ax=ax)
    
            ax.set_title(f'Distribución de {col} por Cluster')
            ax.set_xlabel("Cluster")
            ax.set_ylabel(col)
            ax.legend()
    
            st.pyplot(fig)
    

    else:
        st.warning(f"No se encontró el archivo de datos sintéticos en: {csv_path}")

# ===================================================
# COLUMNA DERECHA: DESCRIPCIÓN Y CONSULTA AL MODELO 
# ===================================================

with col2:
  st.markdown("## 4. Descripción del Proyecto y Motivación")
  st.markdown("""
    Este proyecto utiliza un **modelo de clustering K-Means** para analizar datos obtenidos
    desde sensores **FLC (Fluxgate)** que miden la distorsión magnética del terreno.
    El propósito es **agrupar las observaciones** según sus características físicas y del suelo,
    permitiendo detectar patrones ocultos que podrían estar asociados a distintos **tipos de minas**.""")
    
  st.markdown("### 4.1 Objetivo del Clustering")
  st.markdown(""" 
              El propósito del modelo es **identificar grupos naturales** de observaciones
              sin conocer las etiquetas de salida (M).  
              Mediante clustering se pueden:
                  - Explorar **patrones magnéticos y de suelo**.  
                  - **Distinguir zonas con comportamiento similar.**  
                  - Mejorar la comprensión de los sensores antes de aplicar modelos supervisados.""")

  st.markdown("---")
  st.markdown("### 4.2 Consulta del Modelo de Clustering")
  st.markdown("""Utilizamos los datos originales para realizar un análisis de clustering, identificando las características que distinguen naturalmente a cada grupo.
                 Posteriormente, guardamos el modelo K-Means entrenado y lo aplicamos a una muestra de datos sintéticos para predecir su clasificación.
                 Los resultados se presentan a continuación.""")

    ## Cargar modelo y datos 
  if os.path.exists(model_path) and os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    kmeans = joblib.load(model_path)

    #Intentar cargar el scaler
    if os.path.exists(scaler_path):
      scaler = joblib.load(scaler_path)
      X_scaled = scaler.transform(df[['V','H','S']])
    else:
      X_scaled = df[['V','H','S']].values
      

    #Predicción de clusters 
    df['Cluster_pred'] = kmeans.predict(X_scaled) + 1

    st.write("### 4.3 Clusters asignados por el modelo:")
    st.dataframe(df[['V', 'H', 'S', 'Cluster_pred']].head(10))

    st.write("### 4.4 Distribución de Clusters:")
    st.bar_chart(df['Cluster_pred'].value_counts().sort_index())

    #Botón para descargar resultados 
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label = "Descargar resultados con Clusters",
        data = csv,
        file_name = 'synthetic_clusters.csv',
        mime = "text/csv"
      )
    
    st.markdown("""## 5. Ejemplos de minas terrestres""")
    st.markdown(""" 
                Las siguientes imágenes muestran ejemplos representativos de los **tipos de minas** considerados en el conjunto de datos 
                Estas ilustraciones permiten visualizar las diferencias entre un terreno limpio (sin mina) y los distintos tipos de 
                de artefactos explosivos detectados mediante sensores magnéticos FLC.  
                El objetivo del modelo de clasificación es identificar correctamente el tipo de mina a partir de las mediciones del sensor y
                las condiciones del suelo.""")
                
    img1 = Image.open("Foto_minas.jpeg")
    img2 = Image.open("Foto1.jpeg")
    img3 = Image.open("Foto2.jpeg")
    img4 = Image.open("Foto3.jpeg")
    img5 = Image.open("Foto4.jpeg")
    
    st.image(img1, caption="Ejemplos de tipos de minas terrestres", use_container_width=True)
    st.image(img2, caption="Mina anti-tanque en terreno árido", use_container_width=True)
    st.image(img3, caption="Mina anti-personnel", use_container_width=True)
    st.image(img4, caption="Mina M14 anti-personnel", use_container_width=True)
    st.image(img5, caption="Trampa explosiva anti-personnel", use_container_width=True)

  else:
    st.warning("⚠️ No se encontraron los archivos del modelo o los datos.")
    st.text(f"Modelo: {model_path}")
    st.text(f"Datos: {csv_path}")
