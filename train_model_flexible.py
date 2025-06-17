
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
data_dir = "C://Users/CBU/PycharmProjects/PythonProject/augmented_mels_gentle"  # Update this path as needed
img_size = (128, 128)
model_type = "lstm"  # "cnn", "lstm", or "crnn"
epochs = 100
batch_size = 32

# --- LOAD DATA ---
X = []
y = []
class_names = sorted(os.listdir(data_dir))
label_map = {label: idx for idx, label in enumerate(class_names)}

for label in class_names:
    folder = os.path.join(data_dir, label)
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = load_img(path, target_size=img_size)
        arr = img_to_array(img) / 255.0
        X.append(arr)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)
y_cat = to_categorical(y, num_classes=len(class_names))

# --- TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y, random_state=42
)

# --- MODEL SELECTION ---
if model_type == "lstm":
    X_train = X_train[..., 0]
    X_test = X_test[..., 0]
    model = models.Sequential([
        layers.LSTM(128, input_shape=(img_size[0], img_size[1])),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])

elif model_type == "crnn":
    X_train = X_train[..., 0][..., np.newaxis]
    X_test = X_test[..., 0][..., np.newaxis]
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Reshape((32, -1)),
        layers.LSTM(64, return_sequences=False, dropout=0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])

else:  # CNN
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])

# --- COMPILE AND TRAIN ---
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
    verbose=1
)

# --- EVALUATE ---
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Save predictions for confusion matrix
df_preds = pd.DataFrame({
    'y_true': y_true,
    'y_pred': y_pred
})
df_preds.to_csv("lstm_gentle_augmented_predictions.csv", index=False)

# Optional: Print classification report
print(classification_report(y_true, y_pred, target_names=class_names))
K.clear_session()
