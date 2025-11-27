import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data
with open('../data/training_data.pkl', 'rb') as f:
    training_data = pickle.load(f)

print(training_data[0])
print(len(training_data))

# Prepare features and labels
X_user = []
X_game = []
X_cross = []
y = []

for x in training_data:
    # User features
    user_feats = []
    for k, v in x['user_feature'].items():
        if isinstance(v, np.ndarray):
            user_feats.extend(v.tolist())
        else:
            user_feats.append(v)
    X_user.append(user_feats)
    
    # Game features
    game_feats = []
    for k, v in x['game_feature'].items():
        if isinstance(v, np.ndarray):
            game_feats.extend(v.tolist())
        else:
            game_feats.append(v)
    X_game.append(game_feats)
    
    # Cross features
    cross_feats = []
    for k, v in x['cross_feature'].items():
        cross_feats.append(v)
    X_cross.append(cross_feats)
    
    # Label
    y.append(x['label'])

# Combine all features
X = np.concatenate([
    np.array(X_user, dtype=np.float32),
    np.array(X_game, dtype=np.float32),
    np.array(X_cross, dtype=np.float32)
], axis=1)
y = np.array(y, dtype=np.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Feature dimension: {X_train.shape[1]}")
print(f"Positive rate in training: {y_train.mean():.3f}")
print(f"Positive rate in test: {y_test.mean():.3f}")
print()

# Train Logistic Regression
print("Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=10000, random_state=42, class_weight='balanced', solver='liblinear', C=1.0)
lr_model.fit(X_train, y_train)
print("Training complete!")
print()

# Make predictions
y_pred_train = lr_model.predict(X_train)
y_pred_test = lr_model.predict(X_test)

print("Classification Report: Training Set")
print(classification_report(y_train, y_pred_train, target_names=['Not Recommend', 'Recommend']))

print("Evaluation Metrics: Training Set")
train_precision, train_recall, train_f1, train_support = precision_recall_fscore_support(
    y_train, y_pred_train, average=None, labels=[0, 1]
)

print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
print(f"{'Not Recommend':<15} {train_precision[0]:<12.4f} {train_recall[0]:<12.4f} {train_f1[0]:<12.4f} {train_support[0]:<12}")
print(f"{'Recommend':<15} {train_precision[1]:<12.4f} {train_recall[1]:<12.4f} {train_f1[1]:<12.4f} {train_support[1]:<12}")
print(f"{'Accuracy':<15} {'':<12} {'':<12} {accuracy_score(y_train, y_pred_train):<12.4f} {train_support.sum():<12}")
print(f"{'Macro Avg':<15} {precision_score(y_train, y_pred_train, average='macro'):<12.4f} {recall_score(y_train, y_pred_train, average='macro'):<12.4f} {f1_score(y_train, y_pred_train, average='macro'):<12.4f} {train_support.sum():<12}")
print(f"{'Weighted Avg':<15} {precision_score(y_train, y_pred_train, average='weighted'):<12.4f} {recall_score(y_train, y_pred_train, average='weighted'):<12.4f} {f1_score(y_train, y_pred_train, average='weighted'):<12.4f} {train_support.sum():<12}")
print()

print("Confusion Matrix: Training Set")
cm_train = confusion_matrix(y_train, y_pred_train)
print(cm_train)
print()
print(f"True Negatives (TN): {cm_train[0,0]}")
print(f"False Positives (FP): {cm_train[0,1]}")
print(f"False Negatives (FN): {cm_train[1,0]}")
print(f"True Positives (TP): {cm_train[1,1]}")
print()

print("Classification Report: Test Set")
print(classification_report(y_test, y_pred_test, target_names=['Not Recommend', 'Recommend']))

print("Evaluation Metrics: Test Set")
test_precision, test_recall, test_f1, test_support = precision_recall_fscore_support(
    y_test, y_pred_test, average=None, labels=[0, 1]
)

print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
print(f"{'Not Recommend':<15} {test_precision[0]:<12.4f} {test_recall[0]:<12.4f} {test_f1[0]:<12.4f} {test_support[0]:<12}")
print(f"{'Recommend':<15} {test_precision[1]:<12.4f} {test_recall[1]:<12.4f} {test_f1[1]:<12.4f} {test_support[1]:<12}")
print(f"{'Accuracy':<15} {'':<12} {'':<12} {accuracy_score(y_test, y_pred_test):<12.4f} {test_support.sum():<12}")
print(f"{'Macro Avg':<15} {precision_score(y_test, y_pred_test, average='macro'):<12.4f} {recall_score(y_test, y_pred_test, average='macro'):<12.4f} {f1_score(y_test, y_pred_test, average='macro'):<12.4f} {test_support.sum():<12}")
print(f"{'Weighted Avg':<15} {precision_score(y_test, y_pred_test, average='weighted'):<12.4f} {recall_score(y_test, y_pred_test, average='weighted'):<12.4f} {f1_score(y_test, y_pred_test, average='weighted'):<12.4f} {test_support.sum():<12}")
print()

print("Confusion Matrix: Test Set")
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)
print()
print(f"True Negatives (TN): {cm_test[0,0]}")
print(f"False Positives (FP): {cm_test[0,1]}")
print(f"False Negatives (FN): {cm_test[1,0]}")
print(f"True Positives (TP): {cm_test[1,1]}")
print()

print("Comparison of Training and Test Set Metrics")
print(f"{'Metric':<25} {'Training':<20} {'Test':<20}")
print(f"{'Accuracy':<25} {accuracy_score(y_train, y_pred_train):<20.4f} {accuracy_score(y_test, y_pred_test):<20.4f}")
print(f"{'Precision (Macro)':<25} {precision_score(y_train, y_pred_train, average='macro'):<20.4f} {precision_score(y_test, y_pred_test, average='macro'):<20.4f}")
print(f"{'Recall (Macro)':<25} {recall_score(y_train, y_pred_train, average='macro'):<20.4f} {recall_score(y_test, y_pred_test, average='macro'):<20.4f}")
print(f"{'F1-Score (Macro)':<25} {f1_score(y_train, y_pred_train, average='macro'):<20.4f} {f1_score(y_test, y_pred_test, average='macro'):<20.4f}")
print(f"{'Precision (Weighted)':<25} {precision_score(y_train, y_pred_train, average='weighted'):<20.4f} {precision_score(y_test, y_pred_test, average='weighted'):<20.4f}")
print(f"{'Recall (Weighted)':<25} {recall_score(y_train, y_pred_train, average='weighted'):<20.4f} {recall_score(y_test, y_pred_test, average='weighted'):<20.4f}")
print(f"{'F1-Score (Weighted)':<25} {f1_score(y_train, y_pred_train, average='weighted'):<20.4f} {f1_score(y_test, y_pred_test, average='weighted'):<20.4f}")
print()

# Plot both confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training confusion matrix
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Recommend', 'Recommend'],
            yticklabels=['Not Recommend', 'Recommend'])
axes[0].set_title('Confusion Matrix - Training Set')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Test confusion matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Not Recommend', 'Recommend'],
            yticklabels=['Not Recommend', 'Recommend'])
axes[1].set_title('Confusion Matrix - Test Set')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()

feature_names = []
# User features
for k in training_data[0]['user_feature'].keys():
    if isinstance(training_data[0]['user_feature'][k], np.ndarray):
        feature_names.extend([f'user_{k}_{i}' for i in range(len(training_data[0]['user_feature'][k]))])
    else:
        feature_names.append(f'user_{k}')
# Game features
for k in training_data[0]['game_feature'].keys():
    if isinstance(training_data[0]['game_feature'][k], np.ndarray):
        feature_names.extend([f'game_{k}_{i}' for i in range(len(training_data[0]['game_feature'][k]))])
    else:
        feature_names.append(f'game_{k}')
# Cross features
for k in training_data[0]['cross_feature'].keys():
    feature_names.append(f'cross_{k}')

coefficients = lr_model.coef_[0]
feature_importance = list(zip(feature_names, coefficients))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"{'Rank':<6} {'Feature':<45} {'Coefficient':<15}")
for i, (name, coef) in enumerate(feature_importance[:20], 1):
    print(f"{i:<6} {name:<45} {coef:>15.6f}")