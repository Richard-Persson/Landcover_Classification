import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


# Load trained model
def load_model():
    return tf.keras.models.load_model("models/landcover_cnn.h5")


model = load_model()

# Define class labels
CLASS_LABELS = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

# Sidebar - Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Confusion Matrix", "Upload & Predict"])

# ========================== Confusion Matrix Page ==========================
if page == "Confusion Matrix":
    st.title("Confusion Matrix & Model Performance")

    # Load precomputed test labels & predictions
    y_true = np.load("models/y_true.npy")   # Ground truth labels
    y_pred = np.load("models/y_pred.npy")   # Predicted labels

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    # Show classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_LABELS, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write("### Classification Report")
    st.dataframe(report_df)

# ========================== Image Upload & Prediction Page ==========================
elif page == "Upload & Predict":
    st.title("Upload an Image for Prediction")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_file).resize((128, 128))
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict class
        prediction = model.predict(image_array)
        predicted_class = CLASS_LABELS[np.argmax(prediction)]

        # Display uploaded image
        st.image(image, caption=f"Predicted Class: {predicted_class}", use_column_width=True)

        # Show confidence scores
        st.write("### Confidence Scores:")
        scores_df = pd.DataFrame(prediction[0], index=CLASS_LABELS, columns=["Confidence"])
        st.dataframe(scores_df)


