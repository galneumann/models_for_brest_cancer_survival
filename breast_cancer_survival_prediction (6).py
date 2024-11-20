# A Comparison of Artificial Neural Network and Decision Trees with Logistic Regression as Classification Models for Breast Cancer Survival
The data set was import from: NIH - National Cancer Institute (SEER)

### Imports
"""

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
from sklearn.tree import export_text
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from sklearn.tree import plot_tree
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split, cross_validate, cross_val_score
from scipy.stats import randint
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from IPython.display import display

"""### Load the data"""

data = pd.read_csv('data.csv')
print(data.head())

"""# Data PreProcessing
### Taking care of missing data
"""

data.replace('?', pd.NA, inplace=True)
data.dropna(inplace=True)
data.shape
data.describe()

"""As we can see there are no missing values, "SEER" datasets shoul'd be clean and without missing values.

###Change the target (Survival_probability) to binary terms and remove "survival_probability" and "censor" from data
Acording to the article, the goal is to predict whether a patient survived or not.


"1" = Survived , "0" = Didn't Survived
"""

data['target'] = (data['survival_probability'] > 0.5).astype(int)
Binary_data = data.copy()

data = data.drop('survival_probability', axis=1)
data = data.drop('censor', axis=1)
data.to_csv('data_01.csv', index=False)

data.head()

"""###Now we will split the data into 80% train and 20% test.
We made sure to preserve target variable distribution using Stratification Parameter.
"""

X = data.drop(columns=['target'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

print(f"Stratification of train set: {y_train.mean()}")
print(f"Stratification of test set: {y_test.mean()}")

"""Exports the sets"""

train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')

"""### Adjust the data into Categorical and Numerical datatypes"""

objectAttributesKey = ["diagnosis", "stage_of_cancer", "treatment_administered"]
integerAttributesKey = ["age"]

data[objectAttributesKey] = data[objectAttributesKey].astype('category')
data[integerAttributesKey] = data[integerAttributesKey].astype(int)

data = pd.get_dummies(data, columns=objectAttributesKey, drop_first=True)

data.to_csv('train_data_01.csv', index=False)

data.head()

"""We will check for the distribution of 'target'"""

train_data['target'].value_counts()

"""### Identify outliers
we'll calculate the 1.5 times the Interquartile Range (IQR) and visualize them using boxplots.
"""

train_data['target'] = train_data['target'].map({1: 'Survived', 0: "Didn't Survived"})

numerical_columns = train_data.select_dtypes(include=['int64', 'float64']).columns
numerical_columns = [col for col in numerical_columns if col != 'target']

sns.set(rc={'figure.figsize': (6, 3)})
sns.set_context("talk")

for column in numerical_columns:
  plt.figure(figsize=(6, 3))
  sns.boxplot(x='target', y=column, data=train_data, orient="v", palette="deep")

  plt.title(f'Boxplot of {column} vs Target', fontsize=16)
  plt.xlabel('target', fontsize=14)
  plt.ylabel(column, fontsize=14)

  plt.show()

"""It appears that there are some patients with more then one outlier, we'll remove them from the data set."""

train_data = pd.read_csv('train_data.csv')

numerical_columns = train_data.select_dtypes(include=['int64', 'float64']).columns
numerical_columns = [col for col in numerical_columns if col != 'target']

outliers_mask = pd.Series([False] * len(train_data))

for column in numerical_columns:
    Q1 = train_data[column].quantile(0.25)
    Q3 = train_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_mask |= (train_data[column] < lower_bound) | (train_data[column] > upper_bound)

train_data = train_data[~outliers_mask]

rows_removed = outliers_mask.sum()
print(f"Number of patients removed: {rows_removed}")
train_data.to_csv('train_data_02.csv')

"""### EDA
###Visualizing Features
Apply transformations based on the features's skewness and remove features that still have significant skewness after the transformations
"""

fig, axes = plt.subplots(10, 4, figsize=(40, 40))

axes = axes.flatten()

for index, columnName in enumerate(train_data.columns[1:]):
    ax = axes[index]
    if train_data[columnName].dtype == 'object':
        sns.countplot(x=columnName, data=train_data, ax=ax)
    else:
        sns.histplot(x=columnName, data=train_data, ax=ax)

    ax.set_title(columnName)

plt.tight_layout()
plt.show()

numerical_features = train_data.select_dtypes(include=['int64', 'float64']).columns[1:]

skewed_features = train_data[numerical_features].skew().sort_values(ascending=False)

skewness_df = pd.DataFrame({'Skew': skewed_features})

print(skewness_df)

"""Now we'll perform Log Transformation for the features that has right skewness and Square Root Transformation to the features that has left skewness."""

# Threshold for skewness
threshold = 0.75

def transform_features(train_data, skewed_features, threshold=0.75):
    changed_features = {}
    for feature in skewed_features.index:
        skewness = skewed_features[feature]
        original_values = train_data[feature].copy()

        if skewness > threshold:
            if (train_data[feature] > 0).all():
                train_data[feature] = np.log1p(train_data[feature])
            else:
                print(f"Skipping {feature} due to negative or zero values in the data.")

        elif skewness < -threshold:
            train_data[feature] = np.sqrt(train_data[feature].abs())

        if not train_data[feature].equals(original_values):
            changed_features[feature] = {
                'original_skewness': skewness,
                'new_skewness': train_data[feature].skew()
            }

    return train_data, changed_features

train_data_transformed, changed_features = transform_features(train_data.copy(), skewed_features, threshold)

if changed_features:
    print("\nFeatures with Skewness Changes:")
    for feature, skewness_info in changed_features.items():
        print(f"Feature: {feature}")
        print(f"  Original Skewness: {skewness_info['original_skewness']}")
        print(f"  New Skewness: {skewness_info['new_skewness']}\n")
else:
    print("No features had significant changes in skewness.")

if changed_features:
    fig, axes = plt.subplots(3, 4, figsize=(40, 40))
    axes = axes.flatten() if len(changed_features) > 1 else [axes]

    for ax, (feature, _) in zip(axes, changed_features.items()):
        sns.histplot(train_data_transformed[feature], ax=ax)
        ax.set_title(f"{feature} (Transformed)")

    plt.tight_layout()
    plt.show()

"""After the transformations, we'll remove the features that still has high skewness"""

final_skewness = train_data_transformed[numerical_features].skew().sort_values(ascending=False)

features_to_remove = final_skewness[final_skewness.abs() > threshold].index

if len(features_to_remove) > 0:
    print(f"\nFeatures to be removed due to high skewness (>{threshold}):")
    for feature in features_to_remove:
        print(f" - {feature}")
else:
    print("\nNo features to remove. All features are within the skewness threshold.")

train_data_cleaned = train_data_transformed.drop(columns=features_to_remove)

print(f"\nNumber of features removed: {len(features_to_remove)}")

train_data_cleaned.to_csv('train_data_03.csv', index=False)

"""### Feature Selection Based on Correlation Matrix
Iterative process where we remove one feature at a time, re-calculate the correlation matrix, and then reassess which features to remove next. This approach can help address any changes in correlation dynamics after each feature is removed.
"""

numerical_data = train_data_cleaned.select_dtypes(include=['float64', 'int64'])

correlation_threshold = 0.8

def find_most_correlated(data, threshold):
    correlation_matrix = data.corr()
    features = correlation_matrix.columns
    for i in range(len(features)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                return (features[i], features[j])
    return None

while True:
    pair = find_most_correlated(numerical_data, correlation_threshold)
    if pair is None:
        break
    feature_to_remove = pair[0]
    numerical_data = numerical_data.drop(columns=[feature_to_remove])
    print(f"Removed {feature_to_remove} due to high correlation with {pair[1]}")

plt.figure(figsize=(20, 20))
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Final Correlation Matrix for Numerical Features')
plt.show()

numerical_data.to_csv('reduced_data.csv', index=False)

"""### Feature Scaling
Standardize or normalize your data especially for ANN.


"""

scaler = StandardScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

"""### Modularized Data Processing Functions for Predictive Models:

"""

def create_target_column(dataset):
    dataset['target'] = (dataset['survival_probability'] > 0.5).astype(int)
    dataset.drop(columns=['survival_probability'], inplace=True)
    return dataset

def get_outliers_mask(dataset):

    numerical_columns = dataset.select_dtypes(include=['int64', 'float64']).columns
    outliers_mask = pd.Series([False] * len(dataset), index=dataset.index)

    for column in numerical_columns:
        Q1 = dataset[column].quantile(0.25)
        Q3 = dataset[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask |= (dataset[column] < lower_bound) | (dataset[column] > upper_bound)

    return outliers_mask

def outliers_removal(X, y):

    outliers_mask = get_outliers_mask(X)
    X = X[~outliers_mask]
    y = y.loc[X.index]
    return X, y

def impute_values(X):

    X.replace('?', pd.NA, inplace=True)
    X.dropna(inplace=True)
    return X

def transform_features(X, skewness_threshold=0.75):
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

    skewed_features = X[numerical_columns].skew().sort_values(ascending=False)

    for feature in skewed_features.index:
        skewness = skewed_features[feature]

        if skewness > skewness_threshold:
            if (X[feature] > 0).all():
                X[feature] = np.log1p(X[feature])
            else:
                print(f"Skipping log transformation for {feature} due to zero or negative values.")
        elif skewness < -skewness_threshold:
            X[feature] = np.sqrt(np.abs(X[feature]))

    return X

def remove_skewed_features(X, skewness_threshold=0.75):
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    skewed_features = X[numerical_columns].skew().sort_values(ascending=False)

    features_to_remove = skewed_features[abs(skewed_features) > skewness_threshold].index

    if len(features_to_remove) > 0:
        X = X.drop(columns=features_to_remove)

    return X

def one_hot_encode(X):
    categorical_columns = ['diagnosis', 'stage_of_cancer', 'treatment_administered']
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    return X

def feature_selection_correlation(X, correlation_threshold=0.8):
    def find_most_correlated(data, threshold):
        correlation_matrix = data.corr()
        features = correlation_matrix.columns
        max_correlation = threshold
        feature_to_remove = None
        for i in range(len(features)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > max_correlation:
                    max_correlation = abs(correlation_matrix.iloc[i, j])
                    feature_to_remove = features[i] if abs(correlation_matrix.iloc[i, j]) >= abs(correlation_matrix.iloc[j, i]) else features[j]
        return feature_to_remove

    while True:
        feature_to_remove = find_most_correlated(X, correlation_threshold)
        if feature_to_remove is None:
            break
        X = X.drop(columns=[feature_to_remove])

    return X

def split_data(dataset):
    X = dataset.drop(columns=['target'])
    y = dataset['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test

def apply_transformations(X, y):
    X, y = outliers_removal(X, y)
    X = impute_values(X)
    X = transform_features(X)
    X = remove_skewed_features(X)
    X = one_hot_encode(X)
    X = feature_selection_correlation(X)

    return X, y

def process_data(dataset):
    dataset = create_target_column(dataset)
    X_train, X_test, y_train, y_test = split_data(dataset)
    X_train, y_train = apply_transformations(X_train, y_train)
    X_test, y_test = apply_transformations(X_test, y_test)
    common_columns = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_columns]
    X_test = X_test[common_columns]

    return X_train, X_test, y_train, y_test

"""### Breakdown of the Functions

1.   Create Target Column and Remove Survival Probability
2.   Outliers Removal: Identifies and removes outliers using the IQR method for all numerical features.
3.   Impute Missing Values: Handles missing values by replacing ? with NaN and dropping rows with missing values.
4.   Transformation for Skewness Correction: Applies log transformation for right-skewed data and square root transformation for left-skewed data.
5.   Remove Highly Skewed Features: Removes features with high skewness that remain even after transformations.
6.   One-Hot Encoding: Converts categorical variables into dummy variables (binary format), excluding the first category to avoid multicollinearity.
7.   Feature Selection Based on Correlation: Identifies and removes features that have a correlation higher than the specified threshold (e.g., 0.8).
8.   Split Data into Train and Test Sets (with Stratification).


"""

td = pd.read_csv('data.csv')
X_train, X_test, y_train, y_test = process_data(td)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
results = {}

"""# Logistic Regression

### Basic Training Function
This function will just train the logistic regression model on the entire training set without any cross-validation.
"""

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

"""### Cross-Validation Training Function
This function will perform cross-validation using 5 folds.
"""

def train_and_evaluate_logistic_regression_with_cv(X_train, y_train, n_splits=5):
    model = LogisticRegression(max_iter=1000, random_state=42)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = []
    for train_index, val_index in kf.split(X_train):
        X_train_k, X_val_k = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_k, y_val_k = y_train.iloc[train_index], y_train.iloc[val_index]
        model.fit(X_train_k, y_train_k)
        results = evaluate_logistic_regression(model, X_val_k, y_val_k, f"Validation Fold {len(cv_results)+1}")
        cv_results.append(results)

    # Calculate average of the results
    average_results = pd.DataFrame(cv_results).mean().to_dict()
    print(f"Average K-Folds results: {average_results}")

    return model, average_results

"""### Evaluation Function
This function evaluates the model and returns the results.


"""

def evaluate_logistic_regression(model, X, y, dataset_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    plt.figure(figsize=(6, 3))
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.plot(fpr, tpr, label=f'{dataset_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, square=True)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    return {'Accuracy': accuracy, 'ROC AUC': roc_auc, 'Sensitivity': sensitivity, 'Specificity': specificity}

initial_lr_model = train_logistic_regression(X_train, y_train)

original_train_results = evaluate_logistic_regression(initial_lr_model, X_train, y_train, "Initial Train Set")

cv_lr_model, cv_train_results = train_and_evaluate_logistic_regression_with_cv(X_train, y_train)

test_results = evaluate_logistic_regression(cv_lr_model, X_test, y_test, "Test Set")

test_results = evaluate_logistic_regression(cv_lr_model, X_test, y_test, "Test Set")

results_df = pd.DataFrame({
    'Logistic Regression': original_train_results,
    'Logistic Regression (CV)': cv_train_results,
    'Logistic Regression (Test)': test_results
}).T

display(results_df)

"""# Decision Tree
###Basic train of the Decision Tree Model
"""

def train_decision_tree(X_train, y_train, criterion='gini', max_depth=3, min_samples_split=10, min_samples_leaf=5):
    dt_model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=42)
    dt_model.fit(X_train, y_train)
    return dt_model

"""### Train the Desicion Tree with 5-Folds Cross-Validation"""

def train_decision_tree_with_cv(X_train, y_train, n_splits=5, criterion='gini', max_depth=3, min_samples_split=10, min_samples_leaf=5):
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=42)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(X_train):
        X_train_k, _ = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_k, _ = y_train.iloc[train_index], y_train.iloc[val_index]
        model.fit(X_train_k, y_train_k)

    return model

"""### Evaluate the Decision Tree Model"""

def visualize_tree(dt_model, feature_names):
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, filled=True, feature_names=feature_names, class_names=['0', '1'], rounded=True)
    plt.title('Decision Tree Visualization')
    plt.show()

def evaluate_decision_tree(model, X, y, dataset_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    plt.figure(figsize=(6, 3))
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.plot(fpr, tpr, label=f'{dataset_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, square=True)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    return {'Accuracy': accuracy, 'ROC AUC': roc_auc, 'Sensitivity': sensitivity, 'Specificity': specificity}

dt_initial = train_decision_tree(X_train, y_train)
initial_train_results_dt = evaluate_decision_tree(dt_initial, X_train, y_train, "Decision Tree - Initial Train Set")
visualize_tree(dt_initial, X_train.columns)

results["Decision Tree - Initial Train Set"] = initial_train_results_dt

results_df = pd.DataFrame({
    'Logistic Regression': original_train_results,
    'Logistic Regression (CV)': cv_train_results,
    'Logistic Regression (Test)': test_results,
    'Decision Tree': initial_train_results_dt,
}).T
display(results_df)

dt_cv = train_decision_tree_with_cv(X_train, y_train)
cv_train_results_dt = evaluate_decision_tree(dt_cv, X_train, y_train, "Decision Tree - CV Train Set")
visualize_tree(dt_cv, X_train.columns)
results["Decision Tree - CV Train Set"] = cv_train_results_dt

results_df = pd.DataFrame({
    'Logistic Regression': original_train_results,
    'Logistic Regression (CV)': cv_train_results,
    'Logistic Regression (Test)': test_results,
    'Decision Tree': initial_train_results_dt,
    'Decision Tree (CV)': cv_train_results_dt
}).T
display(results_df)

test_results_dt = evaluate_decision_tree(dt_cv, X_test, y_test, "Decision Tree - Test Set")
results["Decision Tree - Test Set"] = test_results_dt

results_df = pd.DataFrame({
    'Logistic Regression': original_train_results,
    'Logistic Regression (CV)': cv_train_results,
    'Logistic Regression (Test)': test_results,
    'Decision Tree': initial_train_results_dt,
    'Decision Tree (CV)': cv_train_results_dt,
    'Decision Tree (Test)': test_results_dt
}).T
display(results_df)

"""Simulating CHAID"""

dt_model_gini = train_decision_tree(X_train, y_train, criterion='gini', max_depth=3)
evaluate_decision_tree(dt_model_gini, X_test, y_test, 'Decision Tree - Gini - Limited Depth')
visualize_tree(dt_model_gini, X_train.columns)

dt_model_entropy = train_decision_tree(X_train, y_train, criterion='entropy', max_depth=3)
evaluate_decision_tree(dt_model_entropy, X_test, y_test, 'Decision Tree - Entropy - Limited Depth')
visualize_tree(dt_model_entropy, X_train.columns)

results_df = pd.DataFrame({
    'Logistic Regression': original_train_results,
    'Logistic Regression (CV)': cv_train_results,
    'Logistic Regression (Test)': test_results,
    'Decision Tree': initial_train_results_dt,
    'Decision Tree (CV)': cv_train_results_dt,
    'Decision Tree (Test)': test_results_dt,
    'Decision Tree - Gini - Limited Depth': evaluate_decision_tree(dt_model_gini, X_test, y_test, 'Decision Tree - Gini - Limited Depth'),
    'Decision Tree - Entropy - Limited Depth': evaluate_decision_tree(dt_model_entropy, X_test, y_test, 'Decision Tree - Entropy - Limited Depth')
}).T
display(results_df)

"""# ANN (Artificial Neural Networks)
1.  Model Setup: We'll use MLPClassifier from sklearn, which is a feedforward neural network implementation.
2. Network Architecture: We'll start with a simple architecture (one hidden layer) and gradually increase complexity if needed.
3. Training and Evaluation: The process will be similar to what we've done for Logistic Regression and Decision Trees, using ROC AUC, Sensitivity, and Specificity for evaluation.

Train the Neural Network (ANN)
"""

def train_neural_network(X_train, y_train, hidden_layer_sizes=(100,), activation='relu', max_iter=1000):
    ann_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, random_state=42, verbose=True)
    ann_model.fit(X_train, y_train)
    return ann_model

"""### Train ANN with K-Folds Cross-Validation"""

def train_neural_network_with_kfolds(X, y, n_splits=5, hidden_layer_sizes=(100,), activation='relu', max_iter=1000):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        ann_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, random_state=42, verbose=True)
        ann_model.fit(X_train, y_train)

        models.append(ann_model)

    return models

"""Evaluate the Neural Network

"""

def evaluate_neural_network(ann_model, X_train, y_train, model_name):
    y_pred = ann_model.predict(X_train)
    y_prob = ann_model.predict_proba(X_train)[:, 1]

    roc_auc = roc_auc_score(y_train, y_prob)

    fpr, tpr, _ = roc_curve(y_train, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Neural Network (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Neural Network')
    plt.legend()
    plt.show()

    cm = confusion_matrix(y_train, y_pred)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    accuracy = accuracy_score(y_train, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title('Confusion Matrix - Neural Network')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    return {'Accuracy': accuracy, 'ROC AUC': roc_auc, 'Sensitivity': sensitivity, 'Specificity': specificity}

ann_model = train_neural_network(X_train, y_train, hidden_layer_sizes=(100,), activation='relu', max_iter=1000)
initial_train_results_ann = evaluate_neural_network(ann_model, X_train, y_train, 'Artificial Neural Network')
results["ANN - Initial Train Set"] = initial_train_results_ann


results_df = pd.DataFrame({
    'Logistic Regression': original_train_results,
    'Logistic Regression (CV)': cv_train_results,
    'Logistic Regression (Test)': test_results,
    'Decision Tree - Initial Train Set': initial_train_results_dt,
    'Decision Tree - CV Train Set': cv_train_results_dt,
    'Decision Tree - Test Set': test_results_dt,
    'Decision Tree - Gini - Limited Depth': evaluate_decision_tree(dt_model_gini, X_test, y_test, 'Decision Tree - Gini - Limited Depth'),
    'Decision Tree - Entropy - Limited Depth': evaluate_decision_tree(dt_model_entropy, X_test, y_test, 'Decision Tree - Entropy - Limited Depth'),
    'ANN': initial_train_results_ann
}).T
display(results_df)

models = train_neural_network_with_kfolds(X_train, y_train, hidden_layer_sizes=(100,), activation='relu', max_iter=1000)
cv_train_results_ann = evaluate_neural_network(models[-1], X_train, y_train, 'Artificial Neural Network')
results["ANN - CV Train Set"] = cv_train_results_ann


results_df = pd.DataFrame({
    'Logistic Regression': original_train_results,
    'Logistic Regression (CV)': cv_train_results,
    'Logistic Regression (Test)': test_results,
    'Decision Tree - Initial Train Set': initial_train_results_dt,
    'Decision Tree - CV Train Set': cv_train_results_dt,
    'Decision Tree - Test Set': test_results_dt,
    'Decision Tree - Gini - Limited Depth': evaluate_decision_tree(dt_model_gini, X_test, y_test, 'Decision Tree - Gini - Limited Depth'),
    'Decision Tree - Entropy - Limited Depth': evaluate_decision_tree(dt_model_entropy, X_test, y_test, 'Decision Tree - Entropy - Limited Depth'),
    'ANN': initial_train_results_ann,
    'Ann (CV)': cv_train_results_ann
}).T
display(results_df)

test_results_ann = evaluate_neural_network(models[-1], X_test, y_test, 'ANN - Test Set')
results["ANN - Test Set"] = test_results_ann


results_df = pd.DataFrame({
    'Logistic Regression': original_train_results,
    'Logistic Regression (CV)': cv_train_results,
    'Logistic Regression (Test)': test_results,
    'Decision Tree': initial_train_results_dt,
    'Decision Tree (CV)': cv_train_results_dt,
    'Decision Tree (Test)': test_results_dt,
    'Decision Tree - Gini - Limited Depth': evaluate_decision_tree(dt_model_gini, X_test, y_test, 'Decision Tree - Gini - Limited Depth'),
    'Decision Tree - Entropy - Limited Depth': evaluate_decision_tree(dt_model_entropy, X_test, y_test, 'Decision Tree - Entropy - Limited Depth'),
    'ANN': initial_train_results_ann,
    'ANN (CV)': cv_train_results_ann,
    'ANN (Test)': test_results_ann
}).T

display(results_df)

y_pred_lr = initial_lr_model.predict(X_test)
y_pred_lr_cv = cv_lr_model.predict(X_test)

y_pred_dt = dt_initial.predict(X_test)
y_pred_dt_cv = dt_cv.predict(X_test)

y_pred_dt_gini = dt_model_gini.predict(X_test)
y_pred_dt_entropy = dt_model_entropy.predict(X_test)

y_pred_ann = ann_model.predict(X_test)
y_pred_ann_cv = models[-1].predict(X_test)

precision_scores = {
    "Logistic Regression": precision_score(y_test, y_pred_lr),
    "Logistic Regression - CV": precision_score(y_test, y_pred_lr_cv),
    "Decision Tree": precision_score(y_test, y_pred_dt),
    "Decision Tree - CV": precision_score(y_test, y_pred_dt_cv),
    "Decision Tree - Gini - Limited Depth": precision_score(y_test, y_pred_dt_gini),
    "Decision Tree - Entropy - Limited Depth": precision_score(y_test, y_pred_dt_entropy),
    "Artificial Neural Network": precision_score(y_test, y_pred_ann),
    "Artificial Neural Network - CV": precision_score(y_test, y_pred_ann_cv)
}
model_names = list(precision_scores.keys())
precision_values = list(precision_scores.values())
norm = plt.Normalize(min(precision_values), max(precision_values))
colors = [plt.cm.Greens(norm(value)) for value in precision_values]
plt.figure(figsize=(10, 8))
bar = plt.bar(model_names, precision_values, color=colors)
plt.xlabel('Model')
plt.ylabel('Precision Score')
plt.title('Comparison of Model Precision Scores')
plt.xticks(rotation=45, fontsize=6)
sm = cm.ScalarMappable(cmap=plt.cm.Greens, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='Precision Score')
plt.tight_layout()
plt.show()

from google.colab import drive
drive.mount('/content/drive')

! pwd

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# jupyter nbconvert --to html //content/Breast_Cancer_Survival_Prediction_01.ipynb
