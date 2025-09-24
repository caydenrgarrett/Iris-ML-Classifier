# 🌸 Iris Flower Classification with Machine Learning

This project demonstrates a complete machine learning workflow using the famous Iris flowers dataset. The Iris dataset is considered the "Hello World" of machine learning and statistics because it's widely used for practice, learning, and benchmarking.

We build and evaluate multiple algorithms, compare their performance, and select the best-performing model for flower classification.

## 📊 Dataset Overview

- **Total Samples**: 150
- **Features**:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target Classes (Species)**:
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica
- **Balanced dataset** → 50 samples per species

## 🔧 Installation & Requirements

This project uses Python 3 and the SciPy ecosystem. Install dependencies with:

```bash
pip install numpy scipy matplotlib pandas scikit-learn
```

## 🚀 Workflow

### 1. Load & Explore Data
- Load dataset from CSV
- Summarize shape, sample rows, statistical summaries
- Check class distribution

### 2. Data Visualization
- Box plots to inspect variable distribution
- Histograms to view Gaussian-like distributions
- Scatter matrix to explore relationships between variables

### 3. Model Building
We trained and evaluated 6 models with 10-fold cross-validation:

- Logistic Regression (LR)
- Linear Discriminant Analysis (LDA)
- K-Nearest Neighbors (KNN)
- Classification and Regression Trees (CART)
- Gaussian Naive Bayes (NB)
- Support Vector Machines (SVM)

### 4. Model Comparison
Cross-validation accuracy scores (may vary on re-run):

- LR:   0.9609 (0.0521)
- LDA:  0.9740 (0.0401)
- KNN:  0.9572 (0.0433)
- CART: 0.9572 (0.0433)
- NB:   0.9489 (0.0563)
- **SVM:  0.9840 (0.0321)**

### 5. Final Model & Validation
The chosen model (SVM) was tested on a 20% hold-out validation set:

- **Accuracy**: ~97%
- **Confusion Matrix**:
  ```
  [[11  0  0]
   [ 0 12  1]
   [ 0  0  6]]
  ```

- **Classification Report**:
  ```
                precision    recall  f1-score   support
  Iris-setosa       1.00      1.00      1.00        11
  Iris-versicolor   1.00      0.92      0.96        13
  Iris-virginica    0.86      1.00      0.92         6
  ```

## 📈 Results

SVM outperformed other models with ~98% cross-validation accuracy and ~97% validation accuracy. Strong classification across all 3 species with near-perfect precision, recall, and f1-scores.

## 📖 Key Learnings

- How to load and explore a dataset with pandas
- Use of matplotlib and pandas plotting for data visualization
- Evaluation of multiple ML algorithms using scikit-learn
- Model selection with cross-validation and validation sets

## 📚 References

- [UCI Machine Learning Repository – Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

**🔥 This project is a great baseline ML workflow you can adapt to other datasets and problems.**
