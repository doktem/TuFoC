
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# CONFIG
model_types = ["cnn", "lstm"]
data_sources = {
   # "original": "mel_images",
   # "augmented": "augmented_mels",
    "gently_augmented": "augmented_mels_gentle"
}
img_size = (128, 128)
batch_size = 32
epochs = 100
repeats = 30

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_file = f"model_performance_30runs_{timestamp}.xlsx"
os.makedirs("results", exist_ok=True)
output_path = os.path.join("results", output_file)

log_records = []

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
    y_cat = to_categorical(y, num_classes=len(class_names))
    return X, y, y_cat, class_names

def build_model(model_type, input_shape, num_classes):
        if model_type == "cnn":
            base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
            for layer in base_model.layers[:100]:
                layer.trainable = False
            for layer in base_model.layers[100:]:
                layer.trainable = True
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(0.2)(x)
            output = layers.Dense(num_classes, activation='softmax')(x)
            model = models.Model(inputs=base_model.input, outputs=output)
        elif model_type == "lstm":
            model = models.Sequential([
            layers.LSTM(128, input_shape=(img_size[0], img_size[1])),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
            ])
            return model

# MAIN LOOP
for data_type, data_dir in data_sources.items():
    print(f"Loading {data_type.upper()} dataset...")
    X, y, y_cat, class_names = load_dataset(data_dir)

    for model_type in model_types:
        print(f"\n--- Running {model_type.upper()} on {data_type.upper()} dataset ---")

        for run in range(1, repeats + 1):
            print(f"Run {run}/{repeats}")

            X_train, X_test, y_train, y_test = train_test_split(
            X, y_cat, test_size=0.2, stratify=y, random_state=run
            )

        if model_type == "cnn":
            base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
            for layer in base_model.layers[:100]:
                layer.trainable = False
            for layer in base_model.layers[100:]:
                layer.trainable = True
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(0.2)(x)
            output = layers.Dense(num_classes, activation='softmax')(x)
            model = models.Model(inputs=base_model.input, outputs=output)
        elif model_type == "lstm":
            X_train_input = X_train[..., 0]
            X_test_input = X_test[..., 0]

            model = build_model(model_type, input_shape=X_train_input.shape[1:], num_classes=len(class_names))
            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            model.fit(X_train_input, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size,
            callbacks=[early_stop], verbose=0)

            y_pred = np.argmax(model.predict(X_test_input), axis=1)
            y_true = np.argmax(y_test, axis=1)

            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

            result = {
            "Model": model_type.upper(),
            "Dataset": data_type.upper(),
            "Run": run,
            "Accuracy": round(acc, 4),
            "Macro F1-score": round(f1_macro, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4)
            }

            log_records.append(result)

            # Save incrementally to avoid data loss
            pd.DataFrame(log_records).to_excel(output_path, index=False)

            K.clear_session()

            print(f"Finished all runs. Results saved to: {output_path}")
