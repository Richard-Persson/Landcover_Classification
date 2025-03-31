import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
from streamlit_folium import st_folium
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from dotenv import load_dotenv
import os
import requests
import io
import cv2
import skimage

# Laster inn env fil
load_dotenv()

# Henter ut Google api key
google_maps = os.getenv('GOOGLE_MAPS_API')


# Laster inn modellen, bruker st.cache_resource sånn modellen blir kjørt en gang
@st.cache_resource
def load_model(model):
    if model == "CNN_RGB":
        return tf.keras.models.load_model("models/CNN/landcover_cnn_rgb.h5")
    if model == "CNN_MS":
        return tf.keras.models.load_model("models/CNN/landcover_ms.keras")


# Define class labels
CLASS_LABELS = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

# Velg modell
st.sidebar.title("Model")
valg = st.sidebar.selectbox("", options=["CNN_RGB", "CNN_MS"], index=None, placeholder="Velg modell")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Model Performance", "Upload & Predict", "Maps"])
# ========================== Confusion Matrix ==========================
if page == "Model Performance":
    st.title("Confusion Matrix & Model Performance")
    st.text(f"Valgt modell: {valg}")

    if valg == "CNN_RGB":
        model = load_model("CNN_RGB")
        # Load precomputed test labels & predictions
        y_true = np.load("models/CNN/y_true_rgb.npy")   # Ground truth labels
        y_pred = np.load("models/CNN/y_pred_rgb.npy")   # Predicted labels

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        # Classification report
        report = classification_report(y_true, y_pred, target_names=CLASS_LABELS, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("### Classification Report")
        st.dataframe(report_df)

    if valg == "CNN_MS":
        model = load_model("CNN_MS")
        # Load precomputed test labels & predictions
        y_true = np.load("models/CNN/y_true_ms.npy")   # Ground truth labels
        y_pred = np.load("models/CNN/y_pred_ms.npy")   # Predicted labels

        # TODO Dette er en midlertidig fix, forandre y_true og y_pred før lagring av modell etter trening
        y_true = np.argmax(np.load("models/CNN/y_true_ms.npy"), axis=1)  # Konverterer fra én-hot til labels
        y_pred = np.argmax(np.load("models/CNN/y_pred_ms.npy"), axis=1)  # Henter predikerte klasser
        # ⬆️ ⬆️ ⬆️ ⬆️ ⬆️

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        # Classification report
        report = classification_report(y_true, y_pred, target_names=CLASS_LABELS, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("### Classification Report")
        st.dataframe(report_df)


    # ======================== Opplastning av eget bilde ========================
elif page == "Upload & Predict":
    st.title("Upload an Image for Prediction")
    st.text(f"Valgt modell: {valg}")

    if valg == "CNN_RGB":
        model = load_model("CNN_RGB")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Preprosessering av bildet
        image = Image.open(uploaded_file).resize((128, 128))
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        print("Image shape:", image_array.shape)
        print("Image dtype:", image_array.dtype)
        # Predict class
        prediction = model.predict(image_array)
        predicted_class = CLASS_LABELS[np.argmax(prediction)]

        # Viser bildet som er lastet opp
        st.image(image, caption=f"Predicted Class: {predicted_class}", use_container_width=True)

        # Viser confidence scores
        st.write("### Confidence Scores:")
        scores_df = pd.DataFrame(prediction[0], index=CLASS_LABELS, columns=["Confidence"])
        st.dataframe(scores_df)

    # ======================== Maps & Prediksjon ========================
elif page == "Maps":
    st.text(f"Valgt modell: {valg}")

    # Function to get a Google Maps satellite image
    def get_google_maps_image(lat, lon, zoom=16, size="400x400", maptype="satellite"):
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size}&maptype={maptype}&key={google_maps}"
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            st.error("Failed to load map image")
            return None
        st.title("Choose a spot on the map for prediction")
        st.text(f"Valgt modell: {valg}")

    # Definerer kart
    map = folium.Map(location=[50, 10], zoom_start=4, title='Satelite Map', )

    # Legger til satelittbilde
    tile = folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=False,
        control=True
    ).add_to(map)

    # Legger til kartet i streamlit
    map_data = st_folium(map, height=500, width=700)

    if map_data and "center" in map_data:
        lat, lon = map_data["center"]["lat"], map_data["center"]["lng"]
        st.write(f"Kartets senter: {lat}, {lon}")

    # Knapp for å hente satellittbilde
    if st.button("Hent satellittbilde"):
        map_image = get_google_maps_image(lat, lon, zoom=18, size="800x800")

        model = load_model('CNN_RGB')
        # TODO Fikse bilde som blir lastet inn, modellene bommer veldig??
        if map_image:

            map_image = map_image.convert('RGB')
            map_image = np.array(map_image)

            st.image(map_image, caption="Satellittbilde")
            # Forbered bildet for modellen
            img_resized = cv2.resize(map_image, (128, 128))  # Tilpass størrelse til modellen
            img_array = img_resized / 255.0   # Normaliser
            img_array = np.expand_dims(img_array, axis=0)
            # Last inn modellen

            # Kjør prediksjon
            prediction = model.predict(img_array)
            predicted_class = CLASS_LABELS[np.argmax(prediction)]

            st.write(f"Predikert klasse: {predicted_class}")
