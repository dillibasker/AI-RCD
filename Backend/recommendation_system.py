# recommendation_system.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient
import pickle
from flask import Flask, request, jsonify
from datetime import datetime

# Initialize MongoDB client
client = MongoClient('mongodb+srv://dillibasker1:dilli1488@cluster0.rta6lqb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['recommendation_db']
interactions_collection = db['interactions']
items_collection = db['items']

# Generate sample data (for demonstration)
def generate_sample_data():
    users = [f'user_{i}' for i in range(100)]
    items = [f'item_{i}' for i in range(200)]
    interactions = []
    item_tags = {f'item_{i}': f'tag_{i%10} tag_{(i+1)%10}' for i in range(200)}
    
    # Clear existing data
    interactions_collection.delete_many({})
    items_collection.delete_many({})
    
    # Insert interactions
    for user in users:
        for _ in range(np.random.randint(5, 20)):
            item = np.random.choice(items)
            interactions.append({
                'user_id': user,
                'item_id': item,
                'interaction_type': 'like',
                'timestamp': datetime.now()
            })
    
    interactions_collection.insert_many(interactions)
    
    # Insert items
    items_collection.insert_many([
        {'item_id': k, 'tags': v} for k, v in item_tags.items()
    ])

# Load and preprocess data
def load_data():
    interactions = list(interactions_collection.find())
    items = list(items_collection.find())
    
    interactions_df = pd.DataFrame(interactions)
    items_df = pd.DataFrame(items)
    
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    interactions_df['user_id'] = user_encoder.fit_transform(interactions_df['user_id'])
    interactions_df['item_id'] = item_encoder.fit_transform(interactions_df['item_id'])
    
    return interactions_df, items_df, user_encoder, item_encoder

# Build content-based features
def build_content_features(items):
    tfidf = TfidfVectorizer(max_features=100)
    content_matrix = tfidf.fit_transform(items['tags'])
    return content_matrix, tfidf

# Define the recommendation model
class RecommendationModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim, content_matrix):
        super(RecommendationModel, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)
        self.content_matrix = tf.constant(content_matrix, dtype=tf.float32)
        self.content_dense = tf.keras.layers.Dense(embedding_dim, activation='relu')
        
    def call(self, inputs):
        user_ids, item_ids = inputs
        user_vector = self.user_embedding(user_ids)
        item_vector = self.item_embedding(item_ids)
        content_vector = tf.matmul(self.content_matrix, self.content_dense.weights[0]) + self.content_dense.weights[1]
        content_vector = tf.nn.embedding_lookup(content_vector, item_ids)
        score = tf.reduce_sum(user_vector * (item_vector + content_vector), axis=1)
        return score

# Train the model
def train_model():
    interactions, items, user_encoder, item_encoder = load_data()
    content_matrix, tfidf = build_content_features(items)
    
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    embedding_dim = 50
    
    model = RecommendationModel(num_users, num_items, embedding_dim, content_matrix)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Prepare training data
    X = [interactions['user_id'].values, interactions['item_id'].values]
    y = (interactions['interaction_type'] == 'like').astype(int).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
    
    # Save model and encoders
    model.save('recommendation_model')
    with open('user_encoder.pkl', 'wb') as f:
        pickle.dump(user_encoder, f)
    with open('item_encoder.pkl', 'wb') as f:
        pickle.dump(item_encoder, f)
    with open('tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

# Recommendation function
def get_recommendations(user_id, search_query, top_k=5):
    items = list(items_collection.find())
    items_df = pd.DataFrame(items)
    
    # Load model and encoders
    model = tf.keras.models.load_model('recommendation_model')
    with open('user_encoder.pkl', 'rb') as f:
        user_encoder = pickle.load(f)
    with open('item_encoder.pkl', 'rb') as f:
        item_encoder = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    
    # Encode user
    try:
        user_id_encoded = user_encoder.transform([user_id])[0]
    except:
        return []  # New user, return empty recommendations
    
    # Filter items based on search query
    if search_query:
        search_vector = tfidf.transform([search_query]).toarray()
        item_vectors = tfidf.transform(items_df['tags']).toarray()
        scores = np.dot(item_vectors, search_vector.T).flatten()
        relevant_items = items_df.iloc[np.argsort(scores)[::-1][:50]]  # Top 50 relevant items
    else:
        relevant_items = items_df
    
    item_ids = item_encoder.transform(relevant_items['item_id'])
    user_ids = np.array([user_id_encoded] * len(item_ids))
    
    # Predict scores
    scores = model.predict([user_ids, item_ids])
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    return relevant_items.iloc[top_indices]['item_id'].tolist()

# Flask API
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    search_query = data.get('search_query', '')
    recommendations = get_recommendations(user_id, search_query)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    generate_sample_data()
    train_model()
    app.run(debug=True, port=5000)