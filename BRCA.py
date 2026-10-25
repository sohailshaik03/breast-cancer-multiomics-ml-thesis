from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


data = pd.read_csv(PROJECT_DIR / "data.csv").dropna()
X = data.drop("vital.status", axis=1)
y = data["vital.status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} accuracy: {accuracy:.6f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"{model_name} confusion matrix (rows=true, columns=predicted):")
    print(cm)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[0, 1],
        yticklabels=[0, 1],
    )
    plt.xlabel("Predicted vital.status")
    plt.ylabel("Actual vital.status")
    plt.title(f"{model_name} Confusion Matrix")
    save_plot(f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    return accuracy, report, cm


models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(100,), max_iter=500, random_state=42
    ),
    "SVM": SVC(kernel="linear", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
}

results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    accuracy, report, cm = evaluate_model(model, model_name)
    if isinstance(model, SVC):
        scores = model.decision_function(X_test)
    else:
        positive_class_index = list(model.classes_).index(1)
        scores = model.predict_proba(X_test)[:, positive_class_index]
    fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1)
    auc = roc_auc_score(y_test, scores)

    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    save_plot(f"{model_name.lower().replace(' ', '_')}_roc_curve.png")

    results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "class_0_precision": report["0"]["precision"],
            "class_0_recall": report["0"]["recall"],
            "class_0_f1": report["0"]["f1-score"],
            "class_1_precision": report["1"]["precision"],
            "class_1_recall": report["1"]["recall"],
            "class_1_f1": report["1"]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "roc_auc": auc,
            "true_0_predicted_0": int(cm[0, 0]),
            "true_0_predicted_1": int(cm[0, 1]),
            "true_1_predicted_0": int(cm[1, 0]),
            "true_1_predicted_1": int(cm[1, 1]),
        }
    )

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=results_df, x="model", y="accuracy")
for index, value in enumerate(results_df["accuracy"]):
    ax.text(index, value + 0.01, f"{value:.4f}", ha="center")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)
save_plot("model_accuracy_comparison.png")

best_model = results_df.loc[results_df["accuracy"].idxmax()]
print(
    f"Best accuracy: {best_model['model']} "
    f"({best_model['accuracy']:.6f})"
)
print(f"Saved metrics and plots in {OUTPUT_DIR}/")
