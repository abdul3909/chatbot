# Import required libraries
import pandas as pd
import numpy as np
from pymongo import MongoClient
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from utils.trainingData import training_data

# Load the dataset
df = pd.read_csv('dataset/PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv')

# Data preprocessing
# Fix escape sequence and handle out-of-bounds dates
df['Total Price'] = df['Total Price'].replace(r'[\$,]', '', regex=True).astype(float)
df['Creation Date'] = pd.to_datetime(df['Creation Date'], errors='coerce', format='%m/%d/%Y')
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce', format='%m/%d/%Y')
df['Purchase Date'] = df['Purchase Date'].fillna(df['Creation Date'])

# Drop rows with any NaT values after handling dates
df = df.dropna(subset=['Total Price', 'Purchase Date', 'Item Name'])

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['procurement_db']
collection = db['purchases']

# Insert data into MongoDB
data_dict = df.to_dict('records')
collection.delete_many({})  # Clear existing data if any
collection.insert_many(data_dict)

# Separate texts and labels
texts = [text for text, label in training_data]
labels = [label for text, label in training_data]

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define a simple synonym dictionary for preprocessing
synonyms = {
    "spending": "expenditure",
    "items": "products",
    "bought": "purchased",
    "procurement": "orders",
    "expenses": "expenditure",
    "frequently": "often",
    "commodities": "items",
}

# Text preprocessing function with synonym replacement
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = nltk.word_tokenize(text)  # Tokenize
    # Replace synonyms and remove stopwords
    words = [lemmatizer.lemmatize(synonyms.get(word, word)) for word in words if word not in stop_words]
    return ' '.join(words)

# Preprocess the texts
processed_texts = [preprocess_text(text) for text in texts]

# Vectorize texts using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_texts)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X, labels)

# Save the vectorizer and classifier
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(classifier, 'classifier.pkl')
