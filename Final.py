import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request
import matplotlib.pyplot as plt
import seaborn as sns

# Load travel data (replace with your data source)
travel_data = pd.read_csv('travel_data.csv')

# Preprocess the data
def preprocess_text(text):
    # Example preprocessing steps (you can customize these)
    text = text.lower()
    # Additional preprocessing steps...
    return text

travel_data['review'] = travel_data['review'].apply(preprocess_text)

# Exploratory Data Analysis (EDA)
# Example: Plotting average ratings by location
avg_ratings_by_location = travel_data.groupby('location')['rating'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='location', y='rating', data=avg_ratings_by_location)
plt.title('Average Ratings by Location')
plt.xticks(rotation=45)
plt.tight_layout()

# Save EDA plot
plt.savefig('eda_plot.png')

# Initialize NLP components
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(travel_data['review'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on content similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = travel_data[travel_data['location'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 recommendations
    recommended_indices = [i[0] for i in sim_scores]
    return travel_data.iloc[recommended_indices]

# Function for sentiment analysis using NLP
def perform_sentiment_analysis(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

# Initialize Twilio integration
app = Flask(__name__)

@app.route('/sms', methods=['POST'])
def sms():
    user_message = request.form['Body']
    sentiment_score = perform_sentiment_analysis(user_message)
    
    recommended_places = get_recommendations(user_message)

    # Save recommended places to CSV
    recommended_places[['location', 'place', 'rating', 'type']].to_csv('places_2.csv', index=False)

    response = MessagingResponse()
    response.message("Recommended places saved in places_2.csv.\nSentiment Score: {:.2f}\nHere are some recommendations:\n{}".format(sentiment_score, str(recommended_places)))
    
    return str(response)

if __name__ == '__main__':
    app.run(debug=True)
