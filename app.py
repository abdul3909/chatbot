# Import required libraries
from flask import Flask, render_template, request  # Flask for web framework
import joblib  # joblib for loading trained model files
from pymongo import MongoClient  # MongoDB client for data handling
import nltk
from nltk.stem import WordNetLemmatizer  # Lemmatizer for word normalization
from nltk.corpus import stopwords  # Stopwords for filtering out common words
import string  # String module for punctuation removal

# Initialize Flask app
app = Flask(__name__)

# Load vectorizer and classifier models from disk
vectorizer = joblib.load('vectorizer.pkl')
classifier = joblib.load('classifier.pkl')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # MongoDB connection at localhost
db = client['procurement_db']  # Access the procurement database
collection = db['purchases']  # Access the purchases collection

# Synonym dictionary for replacing common synonyms
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
    text = text.lower()  # Convert text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = nltk.word_tokenize(text)  # Tokenize text into words
    # Replace synonyms and remove stopwords
    words = [lemmatizer.lemmatize(synonyms.get(word, word)) for word in words if word not in stop_words]
    return ' '.join(words)

# Home route that serves the main chat interface
@app.route('/')
def home():
    return render_template('index.html')  # Renders index.html for the chat interface

# Chatbot response route that processes user queries
@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_text = request.form['msg']  # Retrieve user's message from form data
    intent = predict_intent(user_text)  # Predict the intent of the message
    response = generate_response(intent, user_text)  # Generate a response based on intent
    return response  # Return the chatbot's response

def predict_intent(text):
    processed_text = preprocess_text(text)  # Preprocess the text
    vectorized_text = vectorizer.transform([processed_text])  # Vectorize the text
    probabilities = classifier.predict_proba(vectorized_text)  # Get probabilities for each intent
    intent_idx = probabilities.argmax()  # Find the index of the highest probability
    confidence = probabilities[0, intent_idx]  # Get the confidence score

    # Lower confidence threshold to 0.6 for testing purposes
    if confidence < 0.6:  # If confidence is below threshold
        return "I'm not sure I understood that. Could you rephrase?"
    else:
        return classifier.classes_[intent_idx]  # Return predicted intent if confidence is high enough


# Generate response based on the identified intent
def generate_response(intent, text):
    if intent == 'total_orders':
        return handle_total_orders(text)
    elif intent == 'highest_spending_quarter':
        return handle_highest_spending_quarter()
    elif intent == 'frequently_ordered_items':
        return handle_frequently_ordered_items()
    elif intent == "I'm not sure I understood that. Could you rephrase?":  # Fallback response
        return intent
    else:
        return "I'm sorry, I didn't understand your request."  # Default response for unknown intents

# Handle 'total_orders' intent
def handle_total_orders(text):
    # Count total orders from MongoDB collection
    total_orders = collection.count_documents({})
    return f"The total number of orders is {total_orders}."

# Handle 'highest_spending_quarter' intent
def handle_highest_spending_quarter():
    # Aggregate data to find the quarter with the highest spending
    pipeline = [
        {
            '$group': {
                '_id': {
                    'year': {'$year': '$Purchase Date'},
                    'quarter': {'$ceil': {'$divide': [{'$month': '$Purchase Date'}, 3]}}
                },
                'total_spent': {'$sum': '$Total Price'}
            }
        },
        {'$sort': {'total_spent': -1}},  # Sort by highest spending
        {'$limit': 1}  # Get top result
    ]
    result = list(collection.aggregate(pipeline))
    if result:
        # Extract year, quarter, and amount from result
        year = result[0]['_id']['year']
        quarter = result[0]['_id']['quarter']
        amount = result[0]['total_spent']
        return f"The highest spending occurred in Q{quarter} of {year}, totaling ${amount:,.2f}."
    else:
        return "No data available to determine the highest spending quarter."

# Handle 'frequently_ordered_items' intent
def handle_frequently_ordered_items():
    # Aggregate data to find the most frequently ordered items
    pipeline = [
        {
            '$group': {
                '_id': '$Item Name',  # Group by item name
                'count': {'$sum': 1}  # Count occurrences of each item
            }
        },
        {'$sort': {'count': -1}},  # Sort by highest count
        {'$limit': 5}  # Limit to top 5 items
    ]
    results = list(collection.aggregate(pipeline))
    # Format the results as a list of items with count
    items = [f"{res['_id']} ({res['count']} times)" for res in results]
    return "The most frequently ordered items are: " + ', '.join(items) + "."

# Run the app on port 4000 in debug mode
if __name__ == "__main__":
    app.run(debug=True, port=4000)
