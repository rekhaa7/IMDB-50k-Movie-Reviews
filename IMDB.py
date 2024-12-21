# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
import re
import string
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from wordcloud import WordCloud

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Set the path to the dataset
import os
os.chdir("/Users/rekhashrestha/Documents/BDA/CSV files")

# Loading the dataset
df = pd.read_csv("IMDB Dataset.csv")

# Dataset properties and cleaning
df.shape
df.isnull().sum()
df.duplicated().sum()

# Removing duplicates
df = df.drop_duplicates()
df.shape

# Dataset preview
df.head()
df.describe()

# Distribution of positive and negative reviews
sns.countplot(x='sentiment', data=df, palette={'positive': 'green', 'negative': 'orange'})
plt.title('Distribution of Positive and Negative Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.show()

# Percentage distribution of sentiments
distribution = df['sentiment'].value_counts(normalize=True) * 100
distribution

# Example reviews
df['review'][0:5]

# Data Preprocessing
def remove_tag(text):
    pattern = re.compile('<[^>]+>')
    return pattern.sub(r'', text)

def remove_urls(text):
    pattern = re.compile(r'\b(?:https?|ftp|www)\S+\b')
    return pattern.sub(r'', text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    new_text = [word for word in text.split() if word not in stop_words]
    return ' '.join(new_text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Cleaning the reviews
df['review_cleaned'] = df['review'].str.lower()
df['review_cleaned'] = df['review_cleaned'].apply(remove_tag)
df['review_cleaned'] = df['review_cleaned'].apply(remove_urls)
df['review_cleaned'] = df['review_cleaned'].apply(remove_stopwords)
df['review_cleaned'] = df['review_cleaned'].apply(remove_punctuation)

# Adding Polarity and Tokenized Words
df['polarity'] = df['review_cleaned'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['tokens'] = df['review_cleaned'].apply(lambda x: TextBlob(x).words)

# Display cleaned dataset
df.head()

# Plot Polarity Distribution
sns.histplot(x='polarity', data=df, kde=True, bins=30)
plt.title('Sentiment Polarity Distribution')
plt.xlabel('Polarity')
plt.ylabel('Frequency')
plt.show()

# Word Clouds for Positive and Negative Reviews
positive_reviews = " ".join(df[df['sentiment'] == 'positive']['review_cleaned'])
negative_reviews = " ".join(df[df['sentiment'] == 'negative']['review_cleaned'])

positive_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(positive_reviews)
negative_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_reviews)

# Display the word clouds
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(positive_wc, interpolation='bilinear')
plt.title('Positive Reviews Word Cloud', fontsize=18)
plt.axis('off')
plt.show()

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 2)
plt.imshow(negative_wc, interpolation='bilinear')
plt.title('Negative Reviews Word Cloud', fontsize=18)
plt.axis('off')

plt.show()

# Preparing data for model training
X = df['review_cleaned']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Extraction using CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression Model with Hyperparameter Tuning
param_grid_lr = {'C': [0.1, 1, 10], 'max_iter': [100, 200, 500]}
grid_search_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train_vec, y_train)
best_lr = grid_search_lr.best_estimator_
print("Best Logistic Regression Parameters:", grid_search_lr.best_params_)

# Random Forest Model with Hyperparameter Tuning
param_grid_rf = {'n_estimators': [100, 200, 500], 'max_depth': [10, 20, None]}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train_vec, y_train)
best_rf = grid_search_rf.best_estimator_
print("Best Random Forest Parameters:", grid_search_rf.best_params_)

# Naive Bayes Model with Hyperparameter Tuning
param_grid_nb = {'alpha': [0.1, 0.5, 1, 5, 10]}
grid_search_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=5, scoring='accuracy')
grid_search_nb.fit(X_train_vec, y_train)
best_nb = grid_search_nb.best_estimator_
print("Best Naive Bayes Parameters:", grid_search_nb.best_params_)

# Evaluate Models
models = {'Logistic Regression': best_lr, 'Random Forest': best_rf, 'Naive Bayes': best_nb}
for model_name, model in models.items():
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print(f"{model_name} Error Rate: {error_rate:.2f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Cross-Validation Results
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
    print(f"{model_name} CV Mean Accuracy: {cv_scores.mean():.2f}")

# Compare Model Performance in a Plot
accuracy_scores = {
    'Logistic Regression': accuracy_score(y_test, best_lr.predict(X_test_vec)),
    'Random Forest': accuracy_score(y_test, best_rf.predict(X_test_vec)),
    'Naive Bayes': accuracy_score(y_test, best_nb.predict(X_test_vec))
}
# Create a DataFrame for plotting
accuracy_df = pd.DataFrame(list(accuracy_scores.items()), columns=['Model', 'Accuracy'])

# Plotting the comparison
plt.figure(figsize=(8, 6))
sns.barplot(x='Accuracy', y='Model', data=accuracy_df, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.show()