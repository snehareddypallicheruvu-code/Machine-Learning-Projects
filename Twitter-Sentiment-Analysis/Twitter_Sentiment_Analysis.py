# TWITTER SENTIMENT ANALYSIS
# ALGORITHM COMPARISON

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. LOAD DATASET

train_path = r"C:\Users\sneha\OneDrive\Documents\ML PROJECT DATASETS\Twitter Sentiment Analysis\twitter_training.csv"

validation_path = r"C:\Users\sneha\OneDrive\Documents\ML PROJECT DATASETS\Twitter Sentiment Analysis\twitter_validation.csv"

train_data = pd.read_csv(train_path, header=None)
validation_data = pd.read_csv(validation_path, header=None)

columns = ['ID', 'Entity', 'Sentiment', 'Tweet']

train_data.columns = columns
validation_data.columns = columns

print("Training Dataset Shape:", train_data.shape)
print("Validation Dataset Shape:", validation_data.shape)

# 2. DATA PREPROCESSING

train_data.dropna(subset=['Sentiment', 'Tweet'], inplace=True)
validation_data.dropna(subset=['Sentiment', 'Tweet'], inplace=True)

def clean_text(text):

    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

train_data['Clean_Tweet'] = train_data['Tweet'].apply(clean_text)
validation_data['Clean_Tweet'] = validation_data['Tweet'].apply(clean_text)

train_data = train_data[train_data['Clean_Tweet'] != '']
validation_data = validation_data[validation_data['Clean_Tweet'] != '']

# 3. INPUT AND TARGET

X_train = train_data['Clean_Tweet']
y_train = train_data['Sentiment']

X_test = validation_data['Clean_Tweet']
y_test = validation_data['Sentiment']

print("\nSENTIMENT DISTRIBUTION")
print(y_train.value_counts())

# CHECK DUPLICATE TWEETS ACROSS DATASETS

train_tweets = set(X_train)
test_tweets = set(X_test)

overlap = train_tweets.intersection(test_tweets)

print("\nExact Cleaned Tweet Overlap:", len(overlap))

# 4. TF-IDF FEATURE EXTRACTION

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF Training Shape:", X_train_tfidf.shape)
print("TF-IDF Testing Shape:", X_test_tfidf.shape)

# 5. DEFINE ALGORITHMS

algorithms = {

    "Linear SVM": LinearSVC(
        class_weight='balanced',
        random_state=42,
        max_iter=3000
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),

    "Naive Bayes": MultinomialNB(
        alpha=0.1
    )
}

# 6. TRAIN AND COMPARE ALGORITHMS

results = []
trained_models = {}
predictions = {}

for name, model in algorithms.items():

    print("\n================================")
    print("TRAINING:", name)
    print("================================")

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    trained_models[name] = model
    predictions[name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average='weighted',
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average='weighted',
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average='weighted',
        zero_division=0
    )

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1-Score:", round(f1 * 100, 2), "%")

    print("\nClassification Report")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

# 7. ALGORITHM COMPARISON TABLE

comparison = pd.DataFrame(results)

print("\n========== ALGORITHM COMPARISON ==========")

print(comparison.round(4).to_string(index=False))

# 8. FIND BEST ALGORITHM

best = comparison.loc[
    comparison['F1-Score'].idxmax()
]

best_name = best['Algorithm']
best_model = trained_models[best_name]
best_pred = predictions[best_name]

print("\nBEST ALGORITHM:", best_name)
print("BEST ACCURACY:", round(best['Accuracy'] * 100, 2), "%")
print("BEST F1-SCORE:", round(best['F1-Score'] * 100, 2), "%")

# 9. SAVE COMPARISON RESULTS

comparison.to_csv(
    "Twitter_Algorithm_Comparison.csv",
    index=False
)

# 10. COLORFUL ALGORITHM COMPARISON

comparison_plot = comparison.melt(
    id_vars='Algorithm',
    value_vars=[
        'Accuracy',
        'Precision',
        'Recall',
        'F1-Score'
    ],
    var_name='Metric',
    value_name='Score'
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=comparison_plot,
    x='Algorithm',
    y='Score',
    hue='Metric',
    palette='Set2'
)

plt.title(
    "Twitter Sentiment Analysis - Algorithm Comparison",
    fontsize=15,
    fontweight='bold'
)

plt.xlabel("Algorithms")
plt.ylabel("Performance Score")
plt.ylim(0, 1.05)
plt.legend(title="Evaluation Metrics")

plt.tight_layout()
plt.savefig("01_Algorithm_Comparison.png", dpi=300)
plt.show()
plt.close()

# 11. BEST MODEL CONFUSION MATRIX

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    best_pred,
    labels=labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlGnBu',
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.5
)

plt.title(
    best_name + " Confusion Matrix",
    fontsize=15,
    fontweight='bold'
)

plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")

plt.tight_layout()
plt.savefig("02_Best_Model_Confusion_Matrix.png", dpi=300)
plt.show()
plt.close()

# 12. SENTIMENT DISTRIBUTION

plt.figure(figsize=(9, 5))

sns.countplot(
    data=train_data,
    x='Sentiment',
    order=train_data['Sentiment'].value_counts().index,
    hue='Sentiment',
    palette='Set2',
    legend=False
)

plt.title(
    "Sentiment Distribution",
    fontsize=15,
    fontweight='bold'
)

plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")

plt.tight_layout()
plt.savefig("03_Sentiment_Distribution.png", dpi=300)
plt.show()
plt.close()

# 13. NEW TWEET PREDICTION

def predict_sentiment(tweet):

    cleaned = clean_text(tweet)

    tweet_tfidf = tfidf.transform([cleaned])

    return best_model.predict(tweet_tfidf)[0]

sample_tweet = "I really love this game. It is amazing!"

result = predict_sentiment(sample_tweet)

print("\nNEW TWEET PREDICTION")
print("Tweet:", sample_tweet)
print("Predicted Sentiment:", result)

# 14. SAVE BEST MODEL

joblib.dump(
    best_model,
    "Best_Twitter_Sentiment_Model.pkl"
)

joblib.dump(
    tfidf,
    "Twitter_TFIDF_Vectorizer.pkl"
)

print("\nALL ALGORITHMS COMPLETED SUCCESSFULLY!")

print("\nGENERATED FILES:")
print("1. Twitter_Algorithm_Comparison.csv")
print("2. 01_Algorithm_Comparison.png")
print("3. 02_Best_Model_Confusion_Matrix.png")
print("4. 03_Sentiment_Distribution.png")
print("5. Best_Twitter_Sentiment_Model.pkl")
print("6. Twitter_TFIDF_Vectorizer.pkl")
