import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, DMatrix, train as xgb_train
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import joblib

# Load the data from CSV
df = pd.read_csv(r"C:\Users\jaidi\Downloads\Data Set - Sheet1 (2).csv")

# Preprocess the data
label_mapping = {"Incline DB Bench": 0, "Preacher Curls": 1, "4+": 0, "2-3": 1, "0-1": 2}
df.replace(label_mapping, inplace=True)
# Function to remove outliers using IQR for each category
def remove_outliers_iqr(df, category_column):
    cleaned_df = pd.DataFrame()
    for category in df[category_column].unique():
        category_df = df[df[category_column] == category]
        Q1 = category_df.quantile(0.25)
        Q3 = category_df.quantile(0.75)
        IQR = Q3 - Q1
        category_df_out = category_df[~((category_df < (Q1 - 1.9 * IQR)) | (category_df > (Q3 + 1.9 * IQR))).any(axis=1)]
        cleaned_df = pd.concat([cleaned_df, category_df_out])
    return cleaned_df

# Apply the IQR method to remove outliers for each rep range
df_cleaned = remove_outliers_iqr(df, "Category")

# Separate features and target variable
X = df_cleaned.drop("Category", axis=1)
y = df_cleaned["Category"]

# Normalize the numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and validation sets
xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.3, random_state=51)

# Convert data to DMatrix for XGBoost
dtrain = DMatrix(xtrain, label=ytrain)
dval = DMatrix(xval, label=yval)

# Set parameters for GridSearchCV
param_grid = {
    'n_estimators': [5, 10, 15,  35],
    'max_depth': [0, 1, 2],
    'learning_rate': [0.0007, 0.001, 0.002, 0.0015],
    'alpha': [0.5, 1, 1.5, 3],
    'lambda': [0.5, 1, 1.5, 3]
}

# Hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(estimator=XGBClassifier(objective='multi:softprob', num_class=3),
                           param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(xtrain, ytrain)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_}")

# Train the model with the best parameters using xgb_train
best_params = grid_search.best_params_
params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'learning_rate': best_params['learning_rate'],
    'max_depth': best_params['max_depth'],
    'alpha': best_params['alpha'],
    'lambda': best_params['lambda'],
    'eval_metric': 'mlogloss'
}

evals = [(dtrain, 'train'), (dval, 'eval')]
results = {}
model = xgb_train(params, dtrain, num_boost_round=1600, evals=evals, early_stopping_rounds=10, evals_result=results, verbose_eval=True)

# Predict and evaluate the model
train_preds = model.predict(dtrain)
val_preds = model.predict(dval)

# Convert predictions from probabilities to class labels
train_preds = [int(x.argmax()) for x in train_preds]
val_preds = [int(x.argmax()) for x in val_preds]

train_accuracy = accuracy_score(ytrain, train_preds)
val_accuracy = accuracy_score(yval, val_preds)
print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

# Detailed classification report
print("Training Classification Report:")
print(classification_report(ytrain, train_preds))
print("Validation Classification Report:")
print(classification_report(yval, val_preds))

# Confusion Matrix
conf_matrix = confusion_matrix(yval, val_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=df['Category'].unique(), yticklabels=df['Category'].unique())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Accuracy vs Epochs
train_mlogloss = results['train']['mlogloss']
val_mlogloss = results['eval']['mlogloss']
epochs = range(1, len(train_mlogloss) + 1)

plt.figure(figsize=(12, 6))
plt.plot(epochs, train_mlogloss, label='Training Log Loss')
plt.plot(epochs, val_mlogloss, label='Validation Log Loss')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.legend()
plt.title('Log Loss vs Epochs')
plt.show()

# Additional Metrics
precision, recall, f1, _ = precision_recall_fscore_support(yval, val_preds, average='weighted')
mae = mean_absolute_error(yval, val_preds)
mse = mean_squared_error(yval, val_preds)
r2 = r2_score(yval, val_preds)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Binarize the output for ROC Curve
yval_binarized = label_binarize(yval, classes=[0, 1, 2])
val_preds_binarized = label_binarize(val_preds, classes=[0, 1, 2])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(yval_binarized[:, i], val_preds_binarized[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-class')
plt.legend(loc="lower right")
plt.show()

# Precision, Recall, F1 Score vs Epochs
plt.figure(figsize=(12, 6))
plt.plot(epochs, [precision]*len(epochs), label='Precision')
plt.plot(epochs, [recall]*len(epochs), label='Recall')
plt.plot(epochs, [f1]*len(epochs), label='F1 Score')
plt.xlabel('Epochs')
plt.ylabel('Scores')
plt.legend()
plt.title('Precision, Recall, and F1 Score vs Epochs')
plt.show()

# Mean Absolute Error and Mean Squared Error vs Epochs
plt.figure(figsize=(12, 6))
plt.plot(epochs, [mae]*len(epochs), label='Mean Absolute Error')
plt.plot(epochs, [mse]*len(epochs), label='Mean Squared Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.title('Mean Absolute Error and Mean Squared Error vs Epochs')
plt.show()

# Log Loss and R-squared vs Epochs
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_mlogloss, label='Training Log Loss')
plt.plot(epochs, val_mlogloss, label='Validation Log Loss')
plt.plot(epochs, [r2]*len(epochs), label='R-squared')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.title('Log Loss and R-squared vs Epochs')
plt.show()

# Save the model to a file
joblib.dump(model, 'path_to_save_model.pkl')
joblib.dump(scaler, 'path_to_save_scaler.pkl')

print("Model saved successfully.")