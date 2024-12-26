import pandas as pd
import numpy as np
import re
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import nltk
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer

import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

print(f'Tensorflow recognized devices: {tf.config.experimental.list_physical_devices()}')
print(f'Tensorflow recognize cuda: {tf.test.is_built_with_cuda()}')

tf.random.set_seed(42)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
df = pd.read_csv("/root/EsameDeepLearning/downsampled_cleaned_lemmatized_similar_data.csv")
print(df.info())
print()
missing_text_rows = df[df['Text'].isna()]
print(missing_text_rows)
print()
df.dropna(axis=0, how='any', inplace=True)
df.reset_index(drop=True, inplace=True)
missing_text_rows = df[df['Text'].isna()]
print(missing_text_rows)


print(len(df))


maxlen = 7500
max_features = 10000

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['Text'])
sequences = tokenizer.texts_to_sequences(df['Text'])
X = tf.keras.utils.pad_sequences(sequences, maxlen=maxlen)
Y = df["Generated"]

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
print("Train length: " + str(len(X_train)) + " " + str(len(y_train)))
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42, stratify=y_val)
print("Validation length: " + str(len(X_val)) + " " + str(len(y_val)))
print("Test length: " + str(len(X_test)) + " " + str(len(y_test)))


def sequence_model(maxlen, max_words, embed_size, metrics):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(
            input_length=maxlen,
            input_dim=max_words, 
            output_dim=embed_size, 
            trainable=True
        ),
        TransformerBlock(embed_dim=embed_size, num_heads=2, ff_dim=32),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, epsilon=0.01)
    model.compile(
        loss='binary_crossentropy', 
        optimizer=optimizer, 
        metrics=metrics
    )
    return model

METRICS = [
    tf.keras.metrics.AUC(name='roc-auc'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name="recall")
          ]

embed_size = 300
maxlen = 7500

model = sequence_model(maxlen, max_features, embed_size, METRICS)
reduceOnPlateu = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=1, verbose=1, restore_best_weights=True),


hist = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val), 
    callbacks=[earlyStopping, reduceOnPlateu],
    shuffle=False,
    verbose=1
)

def plot(history, *metrics, filename=None):
    n_plots = len(metrics)
    fig, axs = plt.subplots(1, n_plots, figsize=(18, 5))

    hist = history.history

    for ax, metric in zip(axs, metrics):
        ax.plot(np.clip(hist[metric], 0, 1))
        ax.plot(np.clip(hist["val_" + metric], 0, 1))
        ax.legend([metric, "val_" + metric])
        ax.set_title(metric)

    if filename:
        plt.savefig(filename)
    #plt.show()

def plot_test_metrics(results, metrics_names, filename=None):
    n_plots = len(metrics_names)
    fig, axs = plt.subplots(1, n_plots, figsize=(18, 5))

    for ax, metric_name, metric_value in zip(axs, metrics_names, results):
        ax.plot([metric_value], marker='o', label=f"Test {metric_name}", color='blue')
        ax.set_title(f"Test {metric_name}")
        ax.set_xlabel("Metrics")
        ax.set_ylabel(metric_name)
        ax.legend()

    if filename:
        plt.savefig(filename)
    #plt.show()


plot(hist, 'loss', 'roc-auc', 'accuracy', 'precision', "recall", filename='training.png')
plot(hist, 'loss', 'roc-auc', 'accuracy', 'precision', "recall", filename='training.pdf')

results = model.evaluate(X_test, y_test, verbose=1)
metrics_names = ['loss', 'roc-auc', 'accuracy', 'precision', 'recall']
plot_test_metrics(results, metrics_names, filename='test.png')
plot_test_metrics(results, metrics_names, filename='test.pdf')

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")