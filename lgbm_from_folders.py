import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from tqdm import tqdm

# --- SETTINGS ---
ROOT_DIR = r"C:\Users\CBU\PycharmProjects\PythonProject\augmented_mels_gentle"
IMG_SIZE = (128, 128)
N_SPLITS = 10
SEED = 42

# --- Collect image paths and labels ---
image_paths = []
labels = []

for region in os.listdir(ROOT_DIR):
    region_path = os.path.join(ROOT_DIR, region)
    if not os.path.isdir(region_path):
        continue
    for fname in os.listdir(region_path):
        if fname.lower().endswith(".png"):
            image_paths.append(os.path.join(region_path, fname))
            labels.append(region)

print(f"Found {len(image_paths)} images across {len(set(labels))} classes.")

# --- Extract HOG features ---
X_features = []
y_labels = []

print("Extracting features...")
for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
    try:
        img = imread(img_path, as_gray=True)
        img = resize(img, IMG_SIZE)
        features = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        X_features.append(features)
        y_labels.append(label)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

X = np.array(X_features)
y = np.array(y_labels)

# --- SMOTE Oversampling ---
print("Applying SMOTE...")
smote = SMOTE(random_state=SEED)
X_bal, y_bal = smote.fit_resample(X, y)

# --- LightGBM with Cross-Validation ---
print("Running LightGBM with 10-fold CV...")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
accs, precs, recs, f1s = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_bal, y_bal)):
    print(f"\nFold {fold + 1}")
    X_train, X_test = X_bal[train_idx], X_bal[test_idx]
    y_train, y_test = y_bal[train_idx], y_bal[test_idx]

    model = LGBMClassifier(learning_rate=0.1, n_estimators=100, boosting_type='gbdt')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    accs.append(report['accuracy'])
    precs.append(report['macro avg']['precision'])
    recs.append(report['macro avg']['recall'])
    f1s.append(report['macro avg']['f1-score'])

# --- Print Results ---
print("\n=== Final Results (10-fold CV) ===")
print(f"Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
print(f"Recall:    {np.mean(recs):.4f} ± {np.std(recs):.4f}")
print(f"F1-score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
