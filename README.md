# Procurement Chatbot

This project is a prototype chatbot designed for procurement professionals. It uses Natural Language Processing (NLP) to interpret procurement-related queries and retrieves data from a large procurement dataset for the State of California, which is stored in MongoDB. The chatbot can handle various types of questions, including total orders in a specified period, the quarter with the highest spending, and frequently ordered items.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Example Queries](#example-queries)
- [Technologies Used](#technologies-used)

## Features
- **Conversational Interface**: A web-based chat interface built with Flask.
- **NLP-Powered Query Understanding**: Classifies user queries using NLP to understand procurement-related questions.
- **Dynamic Data Retrieval**: Accesses procurement data from MongoDB and generates responses based on intent.
- **Multiple Query Types Supported**:
  - Total number of orders in a given time period (month, quarter, year).
  - Identification of the quarter with the highest spending.
  - Analysis of frequently ordered line items.

## Dataset
This chatbot uses a dataset of large purchases made by the State of California, which is available on [Kaggle](https://www.kaggle.com/). The dataset includes fields like Purchase Date, Total Price, Item Name, Department, etc., and is loaded into MongoDB for efficient data querying.

## Setup

### Prerequisites
- Python 3.8+
- MongoDB installed and running on `localhost:27017`
- Basic knowledge of Flask and Python

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/procurement-chatbot.git
   cd procurement-chatbot
   ```

2. **Set up a virtual environment and install dependencies**:
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. **Load the dataset into MongoDB**:
   - Ensure MongoDB is running.
   - Place the dataset CSV file in the `dataset` folder as `PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv`.
   - Run the `train_model.py` script to load the dataset into MongoDB, preprocess the data, and train the model:
     ```bash
     python train_model.py
     ```

4. **Start the Flask app**:
   ```bash
   python app.py
   ```

5. **Access the chatbot**:
   - Open your browser and go to `http://localhost:4000` to interact with the chatbot.

## Usage
1. **Ask a Question**: Type a procurement-related query into the chatbox, such as:
   - "How many orders did we place last month?"
   - "Which quarter had the highest spending?"
   - "What are the most frequently ordered items?"

2. **Receive a Response**: The chatbot will respond based on the data stored in MongoDB and the query’s intent.

## File Structure
```
procurement-chatbot/
├── app.py                 # Main application and chatbot logic
├── train_model.py         # Data loading, preprocessing, and model training script
├── training_data.py       # Separate file containing training data for intent classification
├── requirements.txt       # Required Python packages
├── dataset/
│   └── PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv  # The procurement dataset
├── templates/
│   └── index.html         # HTML template for the chat interface
└── static/
    └── styles.css         # CSS file for styling the chat interface
```

## Example Queries
Use the following sample queries to test the chatbot’s capabilities:
- **Total Orders**:
  - "How many orders were created in the last month?"
  - "Total number of orders in Q1."
- **Highest Spending Quarter**:
  - "Which quarter had the highest spending?"
  - "Show me the quarter with maximum expenditure."
- **Frequently Ordered Items**:
  - "List the most frequently ordered items."
  - "What items are ordered most often?"

## Technologies Used
- **Python**: Programming language
- **Flask**: Web framework for creating the chat interface
- **MongoDB**: Database for storing and querying procurement data
- **NLTK and Scikit-Learn**: NLP libraries for intent classification and data processing
- **HTML/CSS**: Frontend for the chat interface

## Future Improvements
- Expand the chatbot’s functionality to handle more complex procurement queries.
- Incorporate additional datasets for a broader analysis.
- Improve user interface with more interactive elements and a responsive design.

