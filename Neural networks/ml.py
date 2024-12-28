import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef,
    accuracy_score, precision_score, recall_score, f1_score
)

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Funzione per calcolare le metriche
def compute_metric(metric, y_true, y_pred, average='binary'):
    if metric == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric == 'auc':
        return roc_auc_score(y_true, y_pred)
    elif metric == 'mcc':
        return matthews_corrcoef(y_true, y_pred)
    elif metric == 'precision':
        return precision_score(y_true, y_pred, average=average)
    elif metric == 'recall':
        return recall_score(y_true, y_pred, average=average)
    elif metric == 'f1':
        return f1_score(y_true, y_pred, average=average)

print("Data import")
data = pd.read_csv('downsampled_cleaned_lemmatized_similar_data.csv')

maxlen = 7500
max_features = 10000

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data['Text'])
sequences = tokenizer.texts_to_sequences(data['Text'])
X = tf.keras.utils.pad_sequences(sequences, maxlen=maxlen)
Y = data["Generated"]

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
print("Train length: " + str(len(X_train)) + " " + str(len(y_train)))
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42, stratify=y_val)
print("Validation length: " + str(len(X_val)) + " " + str(len(y_val)))
print("Test length: " + str(len(X_test)) + " " + str(len(y_test)))

# Modelli da usare
models = {
    'RandomForestClassifier': RandomForestClassifier(class_weight='balanced', n_jobs=-1),
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=2000, n_jobs=-1),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVM': SVC(class_weight='balanced', C=1),
}

# Dizionario per salvare le metriche
metrics_list = ['accuracy', 'auc', 'mcc', 'precision', 'recall', 'f1']
class_metrics = {model_name: {'train': {metric: {} for metric in metrics_list}, 
                              'validation': {metric: {} for metric in metrics_list}} 
                 for model_name in models.keys()}


# Addestramento e validazione
for model_name, model in models.items():
    print(f"Training and validation of {model_name}")
    
    # Inizializzazione dei contatori
    for metric in metrics_list:
        for cls in ['class_0', 'class_1', 'macro_avg']:
            class_metrics[model_name]['train'][metric][cls] = 0
            class_metrics[model_name]['validation'][metric][cls] = 0
    
    # Addestramento del modello
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calcolo delle metriche per il training set
    for metric in metrics_list:
        for average, label in zip(['binary', 'binary', 'binary', 'macro', 'macro', 'macro'], 
                                    ['class_0', 'class_1', 'macro_avg']):
            train_metric_value = compute_metric(metric, y_train, y_train_pred, average=average)
            class_metrics[model_name]['train'][metric][label] += train_metric_value
    
    # Calcolo delle metriche per il test set
    for metric in metrics_list:
        for average, label in zip(['binary', 'binary', 'binary', 'macro', 'macro', 'macro'], 
                                    ['class_0', 'class_1', 'macro_avg']):
            validation_metric_value = compute_metric(metric, y_test, y_test_pred, average=average)
            class_metrics[model_name]['validation'][metric][label] += validation_metric_value
    
    print(f"{model_name} - Training completed. Train Metrics: {class_metrics[model_name]['train']}, Validation Metrics: {class_metrics[model_name]['validation']}")

# Salvataggio dei risultati
results_df = pd.DataFrame(
    [(model_name, dataset_type, metric, cls, value) 
     for model_name, dataset_metrics in class_metrics.items()
     for dataset_type, metrics in dataset_metrics.items()
     for metric, class_values in metrics.items()
     for cls, value in class_values.items()],
    columns=['Model', 'Dataset', 'Metric', 'Class', 'Value']
)
results_df.to_csv('./ml_cv_results_train_val.csv', index=False)

print("Results saved to 'ml_cv_results_train_val.csv'")
