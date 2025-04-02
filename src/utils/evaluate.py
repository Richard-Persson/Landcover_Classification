import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.swinT_rgb import SwinTransformer as SwinTransformerRGB, PatchEmbedding as PatchEmbeddingRGB, PatchMerging as PatchMergingRGB
from src.models.swinT_multispectral import SwinTransformer as SwinTransformerMS, PatchEmbedding as PatchEmbeddingMS, PatchMerging as PatchMergingMS
from src.utils.data_loader import get_dataset, CLASS_NAMES, IMG_SIZE  # Load test dataset

# Modeller CNN
model_CNN_RGB = tf.keras.models.load_model("models/CNN/landcover_cnn_rgb.h5")
model_CNN_MS = tf.keras.models.load_model("models/CNN/landcover_ms.keras")

# Modeller ResNet
model_ResNet_RGB = tf.keras.models.load_model("models/ResNet/landcover_resnet_rgb.h5")
model_ResNet_MS = tf.keras.models.load_model("models/ResNet/landcover_resnet_multispectral.h5")

# Modeller EffNet
model_EffNet_RGB = tf.keras.models.load_model("models/EfficientNet/landcover_effnet_rgb.h5")
model_EffNet_MS = tf.keras.models.load_model("models/EfficientNet/landcover_effnet_multispectral.h5")

# Modeller Swin
model_Swin_RGB = tf.keras.models.load_model("models/Swin/landcover_swin_rgb.h5",
                                            custom_objects={
                                            "SwinTransformer": SwinTransformerRGB,
                                            "PatchEmbedding": PatchEmbeddingRGB,
                                            "PatchMerging": PatchMergingRGB,
    }
)
model_Swin_RGB.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
)
model_Swin_MS = tf.keras.models.load_model("models/Swin/landcover_swin_multispectral.h5",
                                           custom_objects={
                                            "SwinTransformer": SwinTransformerMS,
                                            "PatchEmbedding": PatchEmbeddingMS,
                                            "PatchMerging": PatchMergingMS,

    })
model_Swin_MS.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
)
patch_size = (2, 2)
def prepare_swin_input(X):
    X = X / 255.0 
    patches = tf.image.extract_patches(
        images=X,
        sizes=(1, patch_size[0], patch_size[1], 1),
        strides=(1, patch_size[0], patch_size[1], 1),
        rates=(1, 1, 1, 1),
        padding="VALID",
    )
    patch_dim = patches.shape[-1]
    return tf.reshape(patches, (-1, (IMG_SIZE // patch_size[0]) ** 2, patch_dim))


# Load test data
_, X_testRGB, _, y_testRGB = get_dataset(IMG_SIZE, "RGB")
_, X_testMS, _, y_testMS = get_dataset(IMG_SIZE, "MS")
X_testSwin_RGB = prepare_swin_input(X_testRGB)
X_testSwin_MS = prepare_swin_input(X_testMS)

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

# EffNet
loss_EffNet_RGB, accuracy_EffNet_RGB = model_EffNet_RGB.evaluate(X_testRGB, y_testRGB)
print(f"Test Loss EffNet_RGB: {loss_EffNet_RGB:.4f}")
print(f"Test Accuracy EffNet_RGB: {accuracy_EffNet_RGB:.4%}")

loss_EffNet_MS, accuracy_EffNet_MS = model_EffNet_MS.evaluate(X_testMS, y_testMS)
print(f"Test Loss EffNet_MS: {loss_EffNet_MS:.4f}")
print(f"Test Accuracy EffNet_MS: {accuracy_EffNet_MS:.4%}")

#Swin
loss_Swin_RGB, acc_Swin_RGB = model_Swin_RGB.evaluate(X_testSwin_RGB, y_testRGB)
print(f"Test Loss Swin_RGB: {loss_Swin_RGB:.4f}")
print(f"Test Accuracy Swin_RGB: {acc_Swin_RGB:.4%}")

loss_Swin_MS, acc_Swin_MS = model_Swin_MS.evaluate(X_testSwin_MS, y_testMS)
print(f"Test Loss Swin_MS: {loss_Swin_MS:.4f}")
print(f"Test Accuracy Swin_MS: {acc_Swin_MS:.4%}")


# Predikere på RGB testdata
y_pred_CNN_RGB = np.argmax(model_CNN_RGB.predict(X_testRGB), axis=1)
y_true_CNN_RGB = np.argmax(y_testRGB, axis=1)

# ResNet
y_pred_ResNet_RGB = np.argmax(model_ResNet_RGB.predict(X_testRGB), axis=1)
y_true_ResNet_RGB = np.argmax(y_testRGB, axis=1)

# EffNet
y_pred_EffNet_RGB = np.argmax(model_EffNet_RGB.predict(X_testRGB), axis=1)
y_true_EffNet_RGB = np.argmax(y_testRGB, axis=1)

# Swin
y_pred_Swin_RGB = np.argmax(model_Swin_RGB.predict(X_testSwin_RGB), axis=1)
y_true_Swin_RGB = np.argmax(y_testRGB, axis=1)

# Predikere på MS testdata
y_pred_CNN_MS = np.argmax(model_CNN_MS.predict(X_testMS), axis=1)
y_true_CNN_MS = np.argmax(y_testMS, axis=1)

# ResNet
y_pred_ResNet_MS = np.argmax(model_ResNet_MS.predict(X_testMS), axis=1)
y_true_ResNet_MS = np.argmax(y_testMS, axis=1)

# EffNet
y_pred_EffNet_MS = np.argmax(model_EffNet_MS.predict(X_testMS), axis=1)
y_true_EffNet_MS = np.argmax(y_testMS, axis=1)

# Swin
y_pred_Swin_MS = np.argmax(model_Swin_MS.predict(X_testSwin_MS), axis=1)
y_true_Swin_MS = np.argmax(y_testMS, axis=1)


# Confusion matrix RGB
cmRGB = confusion_matrix(y_true_CNN_RGB, y_pred_CNN_RGB)
cm_ResNet_RGB = confusion_matrix(y_true_ResNet_RGB, y_pred_ResNet_RGB)
cmMS = confusion_matrix(y_true_CNN_MS, y_pred_CNN_MS)
cm_ResNet_MS = confusion_matrix(y_true_ResNet_MS, y_pred_ResNet_MS)
cm_EffNet_RGB = confusion_matrix(y_true_EffNet_RGB, y_pred_EffNet_RGB)
cm_EffNet_MS = confusion_matrix(y_true_EffNet_MS, y_pred_EffNet_MS)
cm_Swin_RGB = confusion_matrix(y_true_Swin_RGB, y_pred_Swin_RGB)
cm_Swin_MS = confusion_matrix(y_true_Swin_MS, y_pred_Swin_MS)


def plot_confusion_matrix(cm, model_name, ax):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")


# Create a figure with three confusion matrices
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

plot_confusion_matrix(cmRGB, "CNN RGB", axes[0, 0])
plot_confusion_matrix(cmMS, "CNN MS", axes[0, 1])
plot_confusion_matrix(cm_ResNet_RGB, "ResNet RGB", axes[0, 2])
plot_confusion_matrix(cm_ResNet_MS, "ResNet MS", axes[0, 3])
plot_confusion_matrix(cm_EffNet_RGB, "EffNet RGB", axes[1, 0])
plot_confusion_matrix(cm_EffNet_MS, "EffNet MS", axes[1, 1])
plot_confusion_matrix(cm_Swin_RGB, "Swin RGB", axes[1, 2])
plot_confusion_matrix(cm_Swin_MS, "Swin MS", axes[1, 3])

plt.tight_layout()
plt.show()

class_accuracy_cnn_rgb = np.diag(cmRGB) / np.sum(cmRGB, axis=1)
class_accuracy_cnn_ms = np.diag(cmMS) / np.sum(cmMS, axis=1)
class_accuracy_resnet_rgb = np.diag(cm_ResNet_RGB) / np.sum(cm_ResNet_RGB, axis=1)
class_accuracy_resnet_ms = np.diag(cm_ResNet_MS) / np.sum(cm_ResNet_MS, axis=1)
class_accuracy_effnet_rgb = np.diag(cm_EffNet_RGB) / np.sum(cm_EffNet_RGB, axis=1)
class_accuracy_effnet_ms = np.diag(cm_EffNet_MS) / np.sum(cm_EffNet_MS, axis=1)
class_accuracy_swin_rgb = np.diag(cm_Swin_RGB) / np.sum(cm_Swin_RGB, axis=1)
class_accuracy_swin_ms = np.diag(cm_Swin_MS) / np.sum(cm_Swin_MS, axis=1)

x = np.arange(len(CLASS_NAMES))

bar_width = 0.12
x = np.arange(len(CLASS_NAMES))

plt.figure(figsize=(16, 8))
plt.bar(x - 2.5*bar_width, class_accuracy_cnn_rgb, width=bar_width, label="CNN RGB")
plt.bar(x - 1.5*bar_width, class_accuracy_cnn_ms, width=bar_width, label="CNN MS")
plt.bar(x - 0.5*bar_width, class_accuracy_resnet_rgb, width=bar_width, label="ResNet RGB")
plt.bar(x + 0.5*bar_width, class_accuracy_resnet_ms, width=bar_width, label="ResNet MS")
plt.bar(x + 1.5*bar_width, class_accuracy_effnet_rgb, width=bar_width, label="EffNet RGB")
plt.bar(x + 2.5*bar_width, class_accuracy_effnet_ms, width=bar_width, label="EffNet MS")
plt.bar(x + 3.5*bar_width, class_accuracy_swin_rgb, width=bar_width, label="Swin RGB")
plt.bar(x + 4.5*bar_width, class_accuracy_swin_ms, width=bar_width, label="Swin MS")

# Akse og styling
plt.xticks(ticks=x, labels=CLASS_NAMES, rotation=30, ha="right", fontsize=10)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Class-wise Accuracy Comparison", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.ylim(0, 1.1)
plt.legend(ncol=3)
plt.tight_layout()
plt.show()

# Classification report
print("Classification Report CNN RGB:")
print(classification_report(y_true_CNN_RGB, y_pred_CNN_RGB, target_names=CLASS_NAMES))

print("Classification Report CNN MS:")
print(classification_report(y_true_CNN_MS, y_pred_CNN_MS, target_names=CLASS_NAMES))

print("Classification Report ResNet RGB:")
print(classification_report(y_true_ResNet_RGB, y_pred_ResNet_RGB, target_names=CLASS_NAMES))

print("Classification Report ResNet MS:")
print(classification_report(y_true_ResNet_MS, y_pred_ResNet_MS, target_names=CLASS_NAMES))

print("Classification Report EffNet MS:")
print(classification_report(y_true_EffNet_RGB, y_pred_EffNet_RGB, target_names=CLASS_NAMES))

print("Classification Report EffNet MS:")
print(classification_report(y_true_EffNet_MS, y_pred_EffNet_MS, target_names=CLASS_NAMES))

print("Classification Report Swin RGB:")
print(classification_report(y_true_Swin_RGB, y_pred_Swin_RGB, target_names=CLASS_NAMES))

print("Classification Report Swin MS:")
print(classification_report(y_true_Swin_MS, y_pred_Swin_MS, target_names=CLASS_NAMES))