# PRODIGY_ML_02
 Robot Pointing on a Wall  Circle with gradient Circle Task-02  Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset.
## 
🐾 Task-02: Cat vs Dog Classification Using SVM

This project implements a Support Vector Machine (SVM) to classify grayscale images of cats and dogs from the Kaggle Dogs vs Cats dataset. Images are resized to 64x64 and flattened for training. The model achieves solid accuracy using pixel-level features and demonstrates the power of classical ML techniques in image classification.

### 🔧 Tools & Techniques
- SVM (Scikit-learn)
- Image preprocessing (OpenCV)
- Train-test split & evaluation metrics

### 📁 Dataset
[Kaggle - Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

### 📊 Output
- Accuracy score
- Classification report
- Prediction visualization (optional)

Feel free to fork, explore, and build on top of it!

Thanks for the correction! Let's properly label it as Task-02 and present the full implementation accordingly.

🐾 Task-02: Cat vs Dog Classification Using SVM
🎯 Objective
Implement a Support Vector Machine (SVM) to classify images of cats and dogs using the Kaggle dataset.

📦 Dataset
Source: Kaggle - Dogs vs Cats
After downloading and extracting, you’ll find:
- A folder named train/ containing images like cat.0.jpg, dog.0.jpg, etc.

🧰 Required Libraries
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score



🖼️ Load and Preprocess Images
# 📁 Path to the dataset
data_dir = 'train'  # Update this path if needed

# 🧪 Parameters
img_size = 64  # Resize images to 64x64
X = []
y = []

# 🐱🐶 Load images and assign labels
for img_name in os.listdir(data_dir):
    label = 0 if 'cat' in img_name else 1  # 0 = cat, 1 = dog
    img_path = os.path.join(data_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (img_size, img_size))
    X.append(img.flatten())  # Flatten image to 1D array
    y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# ✂️ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



🧠 Train SVM Classifier
# ⚙️ Create and train the SVM model
model = SVC(kernel='linear')  # Try 'rbf' or 'poly' for nonlinear boundaries
model.fit(X_train, y_train)



📊 Evaluate Model
# 🔮 Make predictions
y_pred = model.predict(X_test)

# 📈 Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))



✅ Summary
This basic SVM model uses grayscale, resized images to classify cats and dogs. For better performance, consider:
- Using RGB images
- Applying PCA to reduce dimensionality
- Switching to deep learning (CNNs) for higher accuracy
Would you like help converting this into a deep learning model next, or packaging it for GitHub with a README and notebook?

