import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Loading and Preprocessing
# Load the dataset
data = pd.read_csv('/kaggle/input/brca-multiomics-tcga/data.csv')


# Handle missing values (drop rows with missing values)
data = data.dropna()

# Separate features and target variable
X = data.drop('vital.status', axis=1)
y = data['vital.status']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    return accuracy

# Step 2: Model Training and Evaluation

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest:")
rf_accuracy = evaluate_model(rf, X_test, y_test)


# Random Forest Accuracy Plot

# Plotting Random Forest accuracy as a bar with value on top
plt.figure(figsize=(6, 6))
rf_accuracy_value = rf_accuracy

# Creating the bar plot
plt.bar(['Random Forest'], [rf_accuracy_value], color='green')

# Annotating the bar with accuracy value
plt.text(0, rf_accuracy_value + 0.01, f'{rf_accuracy_value:.4f}', ha='center', va='bottom', fontsize=14, color='black')

# Adding labels and title
plt.ylim(0, 1)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Random Forest Accuracy', fontsize=14)
plt.show()









# Neural Network
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn.fit(X_train, y_train)
print("Neural Network:")
nn_accuracy = evaluate_model(nn, X_test, y_test)



# Neural Network Accuracy Bar Plot

# Plotting the accuracy of the Neural Network model
plt.figure(figsize=(6, 6))
sns.barplot(x=['Neural Network'], y=[nn_accuracy], color='blue')

# Annotating the bar with accuracy value
plt.text(0, nn_accuracy + 0.01, f'{nn_accuracy:.4f}', ha='center', va='bottom', fontsize=14, color='black')

# Set plot labels and title
plt.ylim(0, 1)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Neural Network Accuracy', fontsize=14)

# Display the plot
plt.show()









# Support Vector Machine (SVM)
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
print("Support Vector Machine (SVM):")
svm_accuracy = evaluate_model(svm, X_test, y_test)


# SVM Accuracy Bar Plot
plt.figure(figsize=(6, 6))
sns.barplot(x=["SVM"], y=[svm_accuracy], palette="viridis")

# Annotating the bar with accuracy value
plt.text(0, svm_accuracy + 0.01, f'{svm_accuracy:.4f}', ha='center', fontsize=12)

# Customizing the plot
plt.ylim(0, 1)
plt.title('SVM Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.show()




# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
print("Gradient Boosting:")
gb_accuracy = evaluate_model(gb, X_test, y_test)



# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
print("Gradient Boosting:")
gb_accuracy = evaluate_model(gb, X_test, y_test)

# Plotting the Gradient Boosting accuracy
plt.figure(figsize=(6, 6))
sns.barplot(x=['Gradient Boosting'], y=[gb_accuracy], palette='viridis')

# Annotating the bar with the accuracy value
plt.text(0, gb_accuracy + 0.01, f'{gb_accuracy:.4f}', ha='center', fontsize=14, color='black')

plt.ylim(0, 1)  # Set y-axis limits to make sure the bar and text fit well
plt.title('Gradient Boosting Model Accuracy', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Model', fontsize=14)
plt.show()








# Step 3: Results and Visualization
# Plotting the results
models = ['Random Forest', 'Neural Network', 'SVM', 'Gradient Boosting']
accuracies = [rf_accuracy, nn_accuracy, svm_accuracy, gb_accuracy]


plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)

# Annotating the bars with accuracy values
for index, value in enumerate(accuracies):
    plt.text(index, value + 0.01, f'{value:.4f}', ha='center')

plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.show()



# Identifying the Best Performing Model
best_model_index = np.argmax(accuracies)
best_model_name = models[best_model_index]
best_model_accuracy = accuracies[best_model_index]

print(f'The best performing model is {best_model_name} with an accuracy of {best_model_accuracy}.')

# Additional metrics and visualizations
def plot_roc_curve(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

print("Random Forest ROC Curve:")
plot_roc_curve(rf, X_test, y_test)

print("Neural Network ROC Curve:")
plot_roc_curve(nn, X_test, y_test)

print("Support Vector Machine (SVM) ROC Curve:")
plot_roc_curve(svm, X_test, y_test)

print("Gradient Boosting ROC Curve:")
plot_roc_curve(gb, X_test, y_test)