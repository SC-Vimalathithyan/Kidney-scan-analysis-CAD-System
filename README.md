#  Kidney CT Image Diagnosis with ResNet50

This project performs automated kidney disease classification using CT scan images. It identifies four classes: **Cyst**, **Normal**, **Stone**, and **Tumor** using a deep learning model built on **ResNet50**.

---

## 📂 Dataset

- Format: Folder structure with images categorized into four classes
- Path example:
```

CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/
├── Cyst/
├── Normal/
├── Stone/
└── Tumor/

````

---

## 🛠️ Features

- ✅ Data preprocessing: duplicate & corrupted image removal
- ✅ Resizing & normalization (224x224)
- ✅ CNN-based classifier with **ResNet50 (transfer learning)**
- ✅ Evaluation: Accuracy, Confusion Matrix, Classification Report
- ✅ Live prediction on new images

---

## 📦 Requirements

```bash
pip install numpy opencv-python tensorflow scikit-learn
````

---

## 🧪 How to Use

### 1. **Preprocess Dataset**

* Removes corrupted or duplicate images.
* Resizes and normalizes all images to (224, 224).

```python
cleaned_files = clean_dataset(image_dir)
preprocessed_images = normalize_resize_images(image_dir, target_size=(224, 224))
```

---

### 2. **Train the Model**

```python
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))
```

---

### 3. **Evaluate the Model**

```python
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

---

### 4. **Make Predictions on New Images**

```python
image = preprocess_image("example_ct.jpg")
probs = predict_image(image)
predicted_class = class_names[np.argmax(probs)]
```

---

## 📈 Results

| Metric     | Value (Sample)             |
| ---------- | -------------------------- |
| Accuracy   | \~92%                      |
| Classes    | Cyst, Normal, Stone, Tumor |
| Model Size | \~100MB                    |

---

## 🤖 Model

* Architecture: `ResNet50 (imagenet weights)`
* Layers added:
  * GlobalAveragePooling2D
  * Dense(256, relu)
  * Dense(4, softmax)

---



## 🙋‍♀️ Author

**Jeevika S** [GitHub - jeevika17](https://github.com/jeevika17)
**Vimalathithyan S C** [GitHub - Vimalathithyan S C](https://github.com/SC-Vimalathithyan)
---


Let me know if you'd like me to:
- Add code snippets to `src/` files (train, evaluate, predict).
- Auto-generate `requirements.txt`.
- Improve visualizations (GradCAM, ROC, etc.).
- Help deploy it via Streamlit or Flask.

Would you like me to turn this into a real file and help you push it to GitHub too?
```


