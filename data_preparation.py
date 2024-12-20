import pandas as pd
import numpy as np
import re
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer

import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("data.csv")
df.dropna(axis=0, how='any', inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
print()
print(df.info())
print()
df = df.dropna(subset=['Text'])
print(df.info())
print()
missing_text_rows = df[df['Text'].isna()]
print(missing_text_rows)

print(f"Maximum sequence length: {df.Text.apply(len).max()}")
print(f"Most frequent sequence length: {df.Text.apply(len).mode()[0]}")
print(f"Mean sequence length: {df.Text.apply(len).mean()}")
print(f"Occurreces of class: {df["Generated"].value_counts()}")

sns.countplot(x=df['Generated'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

df.Text.apply(len).plot(kind='hist', bins=50, title="Histogram of question length")

class_counts = df['Generated'].value_counts()
min_class_size = class_counts.min()

class_0 = df[df['Generated'] == 0].sample(min_class_size, random_state=42)
class_1 = df[df['Generated'] == 1].sample(min_class_size, random_state=42)

print("Class 0")
print(class_0.head())
print("Length: " + str(len(class_0)))
print()
print("Class 1")
print(class_1.head())
print("Length " + str(len(class_1)))

balanced_df = pd.concat([class_0, class_1])
df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

sns.countplot(x='Generated', data=df)
plt.title('Balanced Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')

plt.show()

stop = stopwords.words('english')

def cleaning(text):
    text = re.sub(r'[\W\d]+', ' ', text.lower()).strip()
    text = ' '.join([word for word in text.split() if word not in stop])
    return text

df["Text"] = df["Text"].apply(cleaning)
df.head()

lemmatizer = WordNetLemmatizer()

def lemmatize(text):
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop])
    return text

df["Text"] = df["Text"].apply(lemmatize)
df.head()

missing_text_rows = df[df['Text'].isna()]
print(missing_text_rows)

maxlen = 7500
max_features = 10000
sample_n = 64000
chunk_size = 100
top_k = 50

tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
X_tfidf = tfidf_vectorizer.fit_transform(df['Text'].values)

def top_k_indices(matrix, k):
    indices = []
    for _ in range(k):
        max_index = np.unravel_index(np.argmax(matrix), matrix.shape)
        indices.append(max_index)
        matrix[max_index] = -np.inf 
    return indices

output_csv = "downsampled_cleaned_lemmatized_similar_data.csv"

columns = ['Text', 'Generated']
df_combined = pd.DataFrame(columns=columns)

added_rows = set()

num_chunks = (X_tfidf.shape[0] + chunk_size - 1) // chunk_size

for idx in range(num_chunks):
    print(f"Processing chunk {idx + 1}/{num_chunks}")
    

    start_idx = idx * chunk_size
    end_idx = min((idx + 1) * chunk_size, X_tfidf.shape[0])
    chunk = X_tfidf[start_idx:end_idx]

    similarity_matrix = cosine_similarity(chunk)
    
    # Trova i top_k indici più simili
    top_k_indices_list = top_k_indices(similarity_matrix, top_k)
    
    for i, j in top_k_indices_list:
        selected_row = df.iloc[start_idx + i] 
        text = selected_row['Text']
        generated_label = selected_row['Generated']
        
        new_row = {'Text': text, 'Generated': generated_label}
        
        row_key = (text, generated_label)
        if row_key not in added_rows:
            df_combined = pd.concat([df_combined, pd.DataFrame([new_row])], ignore_index=True)
            added_rows.add(row_key)

    if idx == 0:
        df_combined.to_csv(output_csv, index=False, mode='w', header=True)
    else:
        df_combined.to_csv(output_csv, index=False, mode='a', header=False)

    df_combined = pd.DataFrame(columns=columns)

print("Elaborazione completata e dati salvati nel file CSV.")