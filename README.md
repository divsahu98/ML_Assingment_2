# Dry Bean Variety Classification — ML Assignment 2

End-to-end machine-learning workflow: five classifiers are trained and compared on
the UCI Dry Bean dataset, and an interactive Streamlit app is deployed on Streamlit
Community Cloud to evaluate the saved models on uploaded test data.

- **Live Streamlit app:** https://mlassingment2-2026aj05017.streamlit.app/
- **GitHub repository:** https://github.com/divsahu98/ML_Assingment_2

---

## a. Problem statement

Dry beans are the most widely produced pulse crop, and seed quality strongly
affects yield and market price. Sorting seeds by variety by hand is slow and
inconsistent. The goal of this project is to **automatically identify the variety
of a dry bean grain (7 classes) from 16 geometric shape features** that a computer
vision system extracts from an image of the grain.

This is a **multi-class classification** problem. Five classical ML models —
Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and
a Support Vector Machine — are trained on the same data, evaluated with six
metrics (Accuracy, AUC, Precision, Recall, F1, MCC) and compared to find the most
suitable model for this dataset.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Dry Bean Dataset |
| Source | UCI Machine Learning Repository, id 602 — https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |
| Citation | Koklu, M. & Ozkan, I. A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques.* Computers and Electronics in Agriculture, 174, 105507. |
| Instances | 13,611 grains (13,543 after removing 68 exact duplicate rows) |
| Features | 16 numeric (all continuous), no missing values |
| Target | `Class` — 7 varieties: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |
| Type | Multi-class, moderately imbalanced (BOMBAY ≈ 3.9 % … DERMASON ≈ 26.2 %) |

**Features**

| # | Feature | Meaning |
|---|---|---|
| 1 | Area | Number of pixels inside the bean boundary |
| 2 | Perimeter | Length of the bean boundary |
| 3 | MajorAxisLength | Length of the longest line that can be drawn through the bean |
| 4 | MinorAxisLength | Longest line perpendicular to the major axis |
| 5 | AspectRation | Major axis / minor axis (spelling as in the original dataset) |
| 6 | Eccentricity | Eccentricity of the ellipse with the same moments as the region |
| 7 | ConvexArea | Pixels in the smallest convex polygon containing the bean |
| 8 | EquivDiameter | Diameter of a circle with the same area as the bean |
| 9 | Extent | Bean area / bounding-box area |
| 10 | Solidity | Bean area / convex area (convexity) |
| 11 | roundness | 4π·Area / Perimeter² |
| 12 | Compactness | EquivDiameter / MajorAxisLength |
| 13–16 | ShapeFactor1–4 | Additional shape descriptors derived from the axes and area |

**Pre-processing** (all inside `train_models.py`)

1. Strip whitespace from column names / labels; drop rows with missing values (none) and 68 exact duplicates — removing duplicates *before* the split stops identical rows leaking from train into test.
2. Stratified **80 / 20 train–test split** (`random_state = 2026`) so both parts keep the same class proportions. The 20 % hold-out is saved as `test_data.csv`.
3. `StandardScaler` (fitted on the training data only) for Logistic Regression, KNN, Naive Bayes and SVM. The scaler is stored **inside** each saved `Pipeline`, so the app feeds raw features directly to the model. The Decision Tree is scale-invariant and is not scaled.
4. Hyper-parameters tuned with `GridSearchCV`, 5-fold stratified CV on the **training set only**, scoring = weighted F1.

## c. GitHub repository

https://github.com/divsahu98/ML_Assingment_2

```
ML_Assingment_2/
├── .gitignore
├── app.py                # Streamlit app (loads saved models, never retrains)
├── train_models.py       # Training / evaluation source code
├── ml_utils.py           # Shared metric + path helpers used by both scripts
├── requirements.txt      # Dependencies (sklearn/numpy pinned for safe unpickling)
├── README.md
├── test_data.csv         # 20 % hold-out test split (16 features + Class)
├── data/
│   ├── Dry_Bean_Dataset.csv   # UCI Dry Bean dataset (13,611 rows, used for training)
│   └── Dry_Bean_Dataset.txt   # Original UCI dataset description
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics.csv       # Hold-out metrics of all five models
    └── metadata.json     # Features, classes, best hyper-parameters, versions
```

## d. Models used

All five models are trained on the same training split and evaluated on the same
hold-out test set (`test_data.csv`, 2,709 rows). Because the problem is
multi-class, **Precision, Recall and F1 are support-weighted averages** over the 7
classes, **AUC is the One-vs-Rest ROC-AUC (weighted)** computed from
`predict_proba`, and **MCC** is the multi-class Matthews correlation coefficient.

### Comparison table (hold-out test set)

<!-- METRICS_TABLE_START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9258 | 0.9930 | 0.9260 | 0.9258 | 0.9258 | 0.9103 |
| Decision Tree | 0.9110 | 0.9795 | 0.9114 | 0.9110 | 0.9111 | 0.8924 |
| KNN | 0.9247 | 0.9906 | 0.9253 | 0.9247 | 0.9249 | 0.9090 |
| Naive Bayes | 0.8989 | 0.9900 | 0.8996 | 0.8989 | 0.8988 | 0.8781 |
| Support Vector Machine | 0.9280 | 0.9894 | 0.9281 | 0.9280 | 0.9280 | 0.9129 |
<!-- METRICS_TABLE_END -->

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a linear model: 92.58 % accuracy, MCC 0.9103, and the **highest AUC of all models (0.9930)**, so its class probabilities rank samples best. Cross-validation picked C = 1000 (almost no regularisation), which shows that after standard scaling the 7 classes are close to linearly separable in this 16-D shape space. Most of its 201 errors are the DERMASON ↔ SIRA pair (44 + 42). |
| Decision Tree | Lowest AUC (0.9795) and second-lowest accuracy (91.10 %, MCC 0.8924). A tree gives the same probability to every sample in a leaf, so its scores rank samples more coarsely than the other models. Limiting depth to 10 with at least 10 samples per leaf stopped over-fitting: CV F1 (0.9109) ≈ test F1 (0.9111). It is the most interpretable model; ShapeFactor1 (33 %), Compactness (26 %) and MajorAxisLength (19 %) account for most of the splits. SIRA recall is its weakest point (0.852). |
| KNN | 92.47 % accuracy and MCC 0.9090, almost tied with Logistic Regression. Best setting was k = 21 with distance weighting: a fairly large neighbourhood smooths out noise near the class boundaries. Scaling is essential here, otherwise Area (~10⁴–10⁵) would dominate the distance. It has the best HOROZ (0.952) and CALI (0.945) recall. Downside: it stores all 10,834 training rows (largest model file, ~1.3 MB) and is the slowest at prediction time. |
| Naive Bayes | Weakest model overall: 89.89 % accuracy, MCC 0.8781, 274 errors. Gaussian NB assumes the features are independent, but this dataset is highly redundant: 21 feature pairs have \|r\| > 0.9 (e.g. Area–ConvexArea r = 0.9999, Compactness–ShapeFactor3 r = 0.9987), so the same evidence is counted several times and the model becomes over-confident. It confuses DERMASON → SIRA (67) and BARBUNYA → CALI (29) most. Its AUC is still high (0.9900), so the ranking is good even though the hard decisions are worse. It is the fastest model to train. |
| Support Vector Machine | **Best on the threshold metrics**: accuracy 92.80 %, F1 0.9280, MCC 0.9129, and the fewest errors (195). The RBF kernel (C = 10, γ = 0.1) learns slightly non-linear boundaries, which helps on the overlapping DERMASON / SIRA / SEKER region. Its AUC (0.9894) is a little below LR's because SVM probabilities come from an extra Platt (sigmoid) calibration step rather than being produced by the model itself. |
| **Overall, winner for this dataset** | **Support Vector Machine (RBF)**: highest Accuracy, Precision, Recall, F1 and MCC. The margin is small (+0.22 % accuracy over Logistic Regression), and Logistic Regression has the best AUC and is simpler and faster, so it is a strong runner-up if well-ranked probabilities or a lightweight model matter more. For every model BOMBAY is recognised almost perfectly (recall ≈ 1.0, being by far the largest bean), and the main source of error is the visually similar **DERMASON vs SIRA** pair. |

---

## Streamlit app features

| Requirement | Where in the app |
|---|---|
| a. Dataset upload (CSV) | Sidebar → *Upload test CSV*. A sample `test_data.csv` can be downloaded from the sidebar, or the bundled copy can be used with one click. |
| b. Model selection dropdown | Sidebar → *Choose a classifier* (all 5 models) |
| c. Evaluation metrics | *Selected model* tab → Accuracy, AUC, Precision, Recall, F1, MCC |
| d. Confusion matrix / classification report | *Selected model* tab → heat-map (counts or row %) and per-class report |
| Extra | *Compare all models* tab (table + bar chart on the uploaded data), predictions download, dataset & tuned hyper-parameter summary |

The uploaded CSV must contain all 16 feature columns and the `Class` column; extra
columns are ignored and column order does not matter. The app only calls
`predict` / `predict_proba` on the saved pipelines — no retraining happens.

## How to run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
# Install without version pins first: the pinned versions in requirements.txt
# may not exist for an older Python. Training then re-pins requirements.txt to
# the exact versions installed on this machine.
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib openpyxl
python train_models.py                               # trains, saves model/, test_data.csv, updates this table
streamlit run app.py
```

If `data/` is empty, download the dataset first and unzip it into `data/`
(the script finds `Dry_Bean_Dataset.csv` or `Dry_Bean_Dataset.xlsx` in any sub-folder):

```bash
mkdir -p data && cd data
curl -L -o drybean.zip "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip" && unzip -o drybean.zip
cd ..
```

## Deployment (Streamlit Community Cloud)

1. Push this folder to a public GitHub repository (branch `main`).
2. Go to https://streamlit.io/cloud → sign in with GitHub → **New app**.
3. Select the repository, branch `main`, main file `app.py` → **Deploy**.
4. `requirements.txt` pins `scikit-learn` and `numpy` to the versions the models were trained with, because a `.joblib` model saved with one version may fail to load with another.
