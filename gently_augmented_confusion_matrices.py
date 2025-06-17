
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

# Configuration
img_size = (128, 128)
data_dir = "augmented_mels_gentle"
model_types = ["cnn", "lstm"]
num_classes = 5  # Adjust this if your number of classes is different
output_dir = "confusion_matrices"
os.makedirs(output_dir, exist_ok=True)

def load_dataset(data_dir):
    class_names = sorted(os.listdir(data_dir))
    label_map = {label: idx for idx, label in enumerate(class_names)}
    X, y = [], []
    for label in class_names:
        folder = os.path.join(data_dir, label)
        for fname in os.listdir(folder):
            if fname.endswith(".png"):
                path = os.path.join(folder, fname)
                img = load_img(path, target_size=img_size)
                arr = img_to_array(img) / 255.0
                X.append(arr)
                y.append(label_map[label])
    X = np.array(X)
    y = np.array(y)
    return X, y, class_names

X, y, class_names = load_dataset(data_dir)

for model_type in model_types:
    print(f"Processing {model_type.upper()}...")

    if model_type == "cnn":
        X_input = X
        model_path = f"{model_type}_gently_augmented_model.h5"
    elif model_type == "lstm":
        X_input = X[..., 0]
        model_path = f"{model_type}_gently_augmented_model.h5"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        continue

    model = load_model(model_path)
    y_pred = np.argmax(model.predict(X_input), axis=1)

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title(f"{model_type.upper()} Confusion Matrix")
    plt.savefig(os.path.join(output_dir, f"{model_type}_confusion_matrix.png"))
    plt.close()

print("Confusion matrices saved.")
