import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.data_loader import get_dataset, CLASS_NAMES, IMG_SIZE  # Load test dataset

# Modeller CNN
model_CNN_RGB = tf.keras.models.load_model("models/CNN/landcover_cnn_rgb.h5")
model_CNN_MS = tf.keras.models.load_model("models/CNN/landcover_ms.keras")

# Modeller ResNet
model_ResNet_RGB = tf.keras.models.load_model("models/ResNet/landcover_resnet_rgb.h5")
model_ResNet_MS = tf.keras.models.load_model("models/ResNet/landcover_resnet_multispectral.h5")
# Modeller EffNet


# Modeller ....


# Load test data
_, X_testRGB, _, y_testRGB = get_dataset(IMG_SIZE, "RGB")
_, X_testMS, _, y_testMS = get_dataset(IMG_SIZE, "MS")

# CNN
loss_CNN_RGB, accuracy_CNN_RGB = model_CNN_RGB.evaluate(X_testRGB, y_testRGB)
print(f"Test Loss CNN_RGB: {loss_CNN_RGB:.4f}")
print(f"Test Accuracy CNN_RGB: {accuracy_CNN_RGB:.4%}")

loss_CNN_MS, accuracy_CNN_MS = model_CNN_MS.evaluate(X_testMS, y_testMS)
print(f"Test Loss CNN_MS: {loss_CNN_MS:.4f}")
print(f"Test Accuracy CNN_MS: {accuracy_CNN_MS:.4%}")

# ResNet
loss_ResNet_RGB, accuracy_ResNet_RGB = model_ResNet_RGB.evaluate(X_testRGB, y_testRGB)
print(f"Test Loss ResNet_RGB: {loss_ResNet_RGB:.4f}")
print(f"Test Accuracy ResNet_RGB: {accuracy_ResNet_RGB:.4%}")

loss_ResNet_MS, accuracy_ResNet_MS = model_ResNet_MS.evaluate(X_testMS, y_testMS)
print(f"Test Loss ResNet_MS: {loss_ResNet_MS:.4f}")
print(f"Test Accuracy ResNet_MS: {accuracy_ResNet_MS:.4%}")

# Predikere på RGB testdata
y_pred_CNN_RGB = np.argmax(model_CNN_RGB.predict(X_testRGB), axis=1)
y_true_CNN_RGB = np.argmax(y_testRGB, axis=1)

# ResNet
y_pred_ResNet_RGB = np.argmax(model_ResNet_RGB.predict(X_testRGB), axis=1)
y_true_ResNet_RGB = np.argmax(y_testRGB, axis=1)


# Predikere på MS testdata
y_pred_CNN_MS = np.argmax(model_CNN_MS.predict(X_testMS), axis=1)
y_true_CNN_MS = np.argmax(y_testMS, axis=1)

# ResNet
y_pred_ResNet_MS = np.argmax(model_ResNet_MS.predict(X_testMS), axis=1)
y_true_ResNet_MS = np.argmax(y_testMS, axis=1)

# Confusion matrix RGB
cmRGB = confusion_matrix(y_true_CNN_RGB, y_pred_CNN_RGB)
cm_ResNet_RGB = confusion_matrix(y_true_ResNet_RGB, y_pred_ResNet_RGB)
cmMS = confusion_matrix(y_true_CNN_MS, y_pred_CNN_MS)
cm_ResNet_MS = confusion_matrix(y_true_ResNet_MS, y_pred_ResNet_MS)


def plot_confusion_matrix(cm, model_name, ax):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")


# Create a figure with three confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(18, 6))

plot_confusion_matrix(cmRGB, "CNN RGB", axes[0])
plot_confusion_matrix(cmMS, "CNN MS", axes[1])
plot_confusion_matrix(cm_ResNet_RGB, "ResNet RGB", axes[2])
plot_confusion_matrix(cm_ResNet_MS, "ResNet MS", axes[3])

plt.tight_layout()
plt.show()

class_accuracy_cnn_rgb = np.diag(cmRGB) / np.sum(cmRGB, axis=1)
class_accuracy_cnn_ms = np.diag(cmMS) / np.sum(cmMS, axis=1)
class_accuracy_resnet_rgb = np.diag(cm_ResNet_RGB) / np.sum(cm_ResNet_RGB, axis=1)
class_accuracy_resnet_ms = np.diag(cm_ResNet_MS) / np.sum(cm_ResNet_MS, axis=1)

x = np.arange(len(CLASS_NAMES))

bar_width = 0.2

plt.figure(figsize=(12, 6))
plt.bar(x - 1.5*bar_width, class_accuracy_cnn_rgb, width=bar_width, label="CNN RGB", alpha=0.8)
plt.bar(x - 0.5*bar_width, class_accuracy_cnn_ms, width=bar_width, label="CNN MS", alpha=0.8)
plt.bar(x + 0.5*bar_width, class_accuracy_resnet_rgb, width=bar_width, label="ResNet RGB", alpha=0.8)
plt.bar(x + 1.5*bar_width, class_accuracy_resnet_ms, width=bar_width, label="ResNet MS", alpha=0.8)

plt.xticks(ticks=x, labels=CLASS_NAMES, rotation=45)
plt.ylabel("Accuracy")
plt.title("Class-wise Accuracy Comparison")
plt.legend()
plt.show()

# Classification report
print("Classification Report CNN RGB:")
print(classification_report(y_true_CNN_RGB, y_pred_CNN_RGB, target_names=CLASS_NAMES))

print("Classification Report CNN MS:")
print(classification_report(y_true_CNN_MS, y_pred_CNN_MS, target_names=CLASS_NAMES))

print("Classification Report CNN MS:")
print(classification_report(y_true_CNN_MS, y_pred_CNN_MS, target_names=CLASS_NAMES))

print("Classification Report ResNet MS:")
print(classification_report(y_true_ResNet_RGB, y_pred_ResNet_RGB, target_names=CLASS_NAMES))

print("Classification Report ResNet MS:")
print(classification_report(y_true_ResNet_MS, y_pred_ResNet_MS, target_names=CLASS_NAMES))
