"""
train_models.py
---------------
Trains and evaluates the five classifiers required by ML Assignment 2 on the
UCI Dry Bean dataset, then saves everything the Streamlit app needs.

Run (from the project folder):
    python train_models.py
    python train_models.py --data path/to/Dry_Bean_Dataset.xlsx   # custom path

Outputs
    model/*.joblib       one fitted sklearn Pipeline per classifier
    model/metrics.csv    hold-out test metrics for all five models
    model/metadata.json  feature list, classes, best hyper-parameters, versions
    test_data.csv        the untouched 20% hold-out split (features + Class)
    README.md            comparison table between the METRICS markers refreshed
    requirements.txt     pinned to the library versions used for training

The models are trained ONLY here. The app just loads the saved .joblib files
and never retrains on uploaded data.
"""

import argparse
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_utils import (
    DATA_DIR,
    METADATA_PATH,
    METRIC_COLUMNS,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILES,
    PROJECT_DIR,
    TARGET_COL,
    TEST_DATA_PATH,
    compute_metrics,
)

SEED = 2026          # one seed for the split, CV folds and stochastic models
TEST_SIZE = 0.20     # 80% train / 20% hold-out test
CV_FOLDS = 5

README_PATH = PROJECT_DIR / "README.md"
REQUIREMENTS_PATH = PROJECT_DIR / "requirements.txt"
TABLE_START = "<!-- METRICS_TABLE_START -->"
TABLE_END = "<!-- METRICS_TABLE_END -->"


# --------------------------------------------------------------------------- #
# 1. Data loading & cleaning
# --------------------------------------------------------------------------- #
def find_dataset_file():
    """Locate the Dry Bean file inside data/.

    The CSV is preferred (plain pandas, no extra dependency); the original
    .xlsx from the UCI zip is used as a fallback and needs openpyxl.
    """
    for pattern in ("**/Dry_Bean_Dataset.csv", "**/Dry_Bean_Dataset.xlsx", "**/*.csv", "**/*.xlsx"):
        matches = sorted(DATA_DIR.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No dataset found in {DATA_DIR}. Download the Dry Bean dataset from "
        "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset and unzip it into data/."
    )


def load_dataset(path):
    """Read the raw file and apply light, leakage-free cleaning."""
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)

    # Strip stray whitespace from headers and labels so the app's column
    # check does not fail on e.g. "Class " vs "Class".
    df.columns = [str(c).strip() for c in df.columns]
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()

    n_raw = len(df)
    df = df.dropna()
    n_missing = n_raw - len(df)

    # Dry Bean contains a few exact duplicate rows. Dropping them BEFORE the
    # split prevents the same sample appearing in both train and test, which
    # would inflate the test scores.
    n_before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_duplicates = n_before_dedup - len(df)

    print(f"Loaded {path.name}: {n_raw} rows, {df.shape[1] - 1} features")
    print(f"  removed {n_missing} rows with missing values, {n_duplicates} duplicate rows")
    return df, {"raw_rows": n_raw, "missing_rows_removed": n_missing, "duplicates_removed": n_duplicates}


# --------------------------------------------------------------------------- #
# 2. Model definitions
# --------------------------------------------------------------------------- #
def build_model_grid():
    """Return {name: (pipeline, hyper-parameter grid)} for the five models.

    Every model is wrapped in a Pipeline so that the scaler learnt on the
    TRAINING data is saved together with the classifier. The app can then feed
    raw test features straight into the pipeline with no separate
    preprocessing artefact to keep in sync.
    """
    return {
        # Linear model; scaling is required because the features span very
        # different ranges (Area ~ 10^4-10^5, ShapeFactor1 ~ 10^-3).
        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=SEED)),
            ]),
            {"clf__C": [0.1, 1, 10, 100, 1000], "clf__class_weight": [None, "balanced"]},
        ),
        # Trees split on thresholds, so they are scale-invariant: no scaler.
        # max_depth / min_samples_leaf are tuned to control over-fitting.
        "Decision Tree": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [6, 10, 14, None],
                "clf__min_samples_leaf": [1, 5, 10],
            },
        ),
        # Distance based -> scaling is essential, otherwise Area/ConvexArea
        # would dominate the Euclidean distance completely.
        "KNN": (
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [5, 9, 15, 21, 31, 41], "clf__weights": ["uniform", "distance"]},
        ),
        # GaussianNB is chosen over MultinomialNB because the features are
        # continuous real values, not counts. Scaling matters here too:
        # var_smoothing adds (epsilon * largest feature variance) to every
        # variance, and on raw data the largest variance is that of Area
        # (~10^8), which would swamp tiny-variance features like Compactness.
        "Naive Bayes": (
            Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
        ),
        # RBF-kernel SVM. An SVC only outputs decision scores, so probabilities
        # (needed for AUC) are added after tuning with a calibration wrapper -
        # see train_and_evaluate.
        "Support Vector Machine": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", random_state=SEED)),
            ]),
            {"clf__C": [1, 10, 50, 100, 200], "clf__gamma": ["scale", 0.05, 0.1]},
        ),
    }


# --------------------------------------------------------------------------- #
# 3. Training, evaluation, persistence
# --------------------------------------------------------------------------- #
def train_and_evaluate(X_train, y_train, X_test, y_test, n_jobs=-1):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    rows, details = [], {}

    for name, (pipeline, grid) in build_model_grid().items():
        start = time.perf_counter()

        # Hyper-parameters are chosen with cross-validation on the TRAINING set
        # only; the test set is never seen until the final evaluation below.
        search = GridSearchCV(pipeline, grid, cv=cv, scoring="f1_weighted", n_jobs=n_jobs)
        search.fit(X_train, y_train)
        model = search.best_estimator_

        if name == "Support Vector Machine":
            # Wrap the tuned SVC in Platt (sigmoid) calibration so the saved
            # model has predict_proba. SVC(probability=True) is deprecated
            # since scikit-learn 1.9; CalibratedClassifierCV with
            # ensemble=False is the recommended replacement and also works on
            # older versions. A side benefit: predict() is the arg-max of the
            # probabilities, so labels and probabilities never disagree.
            calibrated_svc = CalibratedClassifierCV(
                model.named_steps["clf"], method="sigmoid", cv=cv, ensemble=False
            )
            model.set_params(clf=calibrated_svc).fit(X_train, y_train)

        elapsed = time.perf_counter() - start

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba, model.classes_)
        rows.append({"ML Model Name": name, **metrics})

        joblib.dump(model, MODEL_DIR / MODEL_FILES[name], compress=3)

        # Store the tuned values in plain-Python form for metadata.json.
        best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        details[name] = {
            "best_params": best_params,
            "cv_f1_weighted": round(float(search.best_score_), 4),
            "train_seconds": round(elapsed, 1),
        }
        print(f"  {name:<24} acc={metrics['Accuracy']:.4f}  f1={metrics['F1']:.4f}  "
              f"best={best_params}  ({elapsed:.1f}s)")

    return pd.DataFrame(rows), details


def update_readme(results):
    """Replace the comparison table in README.md between the two markers."""
    if not README_PATH.exists():
        return
    text = README_PATH.read_text(encoding="utf-8")
    if TABLE_START not in text or TABLE_END not in text:
        print("  README markers not found - table not updated")
        return

    header = "| ML Model Name | " + " | ".join(METRIC_COLUMNS) + " |"
    divider = "|---|" + "---|" * len(METRIC_COLUMNS)
    lines = [header, divider]
    for _, row in results.iterrows():
        values = " | ".join(f"{row[m]:.4f}" for m in METRIC_COLUMNS)
        lines.append(f"| {row['ML Model Name']} | {values} |")

    before = text.split(TABLE_START)[0]
    after = text.split(TABLE_END)[1]
    README_PATH.write_text(
        f"{before}{TABLE_START}\n" + "\n".join(lines) + f"\n{TABLE_END}{after}", encoding="utf-8"
    )
    print("  README.md comparison table updated")


def write_requirements():
    """Pin the libraries that matter for unpickling the saved models.

    A .joblib file created with one scikit-learn / NumPy version is not
    guaranteed to load with another, and a mismatch is the most common reason
    a Streamlit Cloud deployment fails. So we pin those to the exact versions
    used here, and keep the plotting / UI libraries as lower bounds.
    """
    def v(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    pinned = ["scikit-learn", "numpy", "pandas", "joblib"]
    # openpyxl is only needed by this script to read the original .xlsx file.
    lower_bound = ["streamlit", "matplotlib", "seaborn", "openpyxl"]
    lines = [f"{p}=={v(p)}" if v(p) else p for p in pinned]
    lines += [f"{p}>={v(p)}" if v(p) else p for p in lower_bound]
    REQUIREMENTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  requirements.txt written: {', '.join(lines)}")


def main():
    parser = argparse.ArgumentParser(description="Train the 5 classifiers for ML Assignment 2")
    parser.add_argument("--data", help="path to Dry_Bean_Dataset.xlsx/.csv (default: search data/)")
    # -1 = use all CPU cores. Use 1 on machines that do not allow
    # multiprocessing (the results are identical, only slower).
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel jobs for GridSearchCV")
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else find_dataset_file()
    df, cleaning = load_dataset(data_path)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    # Assignment constraints: >= 12 features and >= 500 instances.
    assert len(feature_cols) >= 12, f"Need >= 12 features, found {len(feature_cols)}"
    assert len(df) >= 500, f"Need >= 500 instances, found {len(df)}"

    X, y = df[feature_cols], df[TARGET_COL]

    # Stratified split keeps the class proportions identical in train and
    # test, which matters because the classes are imbalanced.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | Classes: {y.nunique()}")

    MODEL_DIR.mkdir(exist_ok=True)

    # Save the hold-out split exactly as the app expects it: features + target.
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_DATA_PATH, index=False)
    print(f"  test data saved to {TEST_DATA_PATH.name}")

    print("Training models (GridSearchCV, 5-fold stratified CV on the training set):")
    results, details = train_and_evaluate(X_train, y_train, X_test, y_test, n_jobs=args.n_jobs)

    results.to_csv(METRICS_PATH, index=False)
    print("\nHold-out test metrics:")
    print(results.set_index("ML Model Name").round(4).to_string())

    metadata = {
        "dataset": "UCI Dry Bean Dataset (id 602)",
        "source_file": data_path.name,
        "target": TARGET_COL,
        "features": feature_cols,
        "classes": sorted(y.unique().tolist()),
        "n_rows_after_cleaning": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_distribution": {k: int(v) for k, v in y.value_counts().sort_index().items()},
        "cleaning": cleaning,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "models": details,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    update_readme(results)
    write_requirements()
    print("\nDone. Launch the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
