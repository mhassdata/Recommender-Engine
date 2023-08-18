# Travel Recommendations with Recommender-Engine & Unsupervised learning 

# GOAL

The goal of this project was to utilize the data sourced in order to create target customers and then take a marketing approach on how to create customer profiles based on these results.

# Unsupervised Travel Recommendations using Google Reviews
## Step 1: Data Collection and Preprocessing
Collect the Google Reviews dataset for activities across 24 categories in European destinations.
Preprocess the text data by removing stopwords, special characters, and perform text normalization.
## Step 2: Feature Extraction
Transform the preprocessed text data into numerical representations using techniques like TF-IDF or word embeddings.
Construct a feature matrix for clustering algorithms.
## Step 3: Clustering
Apply unsupervised clustering algorithms like K-Means, Hierarchical Clustering, or DBSCAN to group similar activities together.
Experiment with different numbers of clusters and algorithms to find the best grouping.
## Step 4: Recommendation Generation
When a user indicates preferences, calculate the similarity between their input and each cluster's centroid.
Recommend top-rated activities within the most similar cluster(s).
# Travel Recommender with Twilio
## Step 1: Set Up Twilio Integration
Create a Twilio account and obtain your API credentials (Account SID, Auth Token, Twilio phone number).
Set up a webhook to receive incoming SMS messages.
## Step 2: Handling Incoming SMS
When Twilio sends an incoming SMS webhook, parse the user's preferences from the message.
Use the unsupervised clustering model to map the user's preferences to relevant clusters.
## Step 3: Generating Recommendations
Retrieve the top-rated activities from the matched cluster(s).
Prepare a response SMS with the recommended activities and send it back to the user via Twilio.

# PowerPoint Presentation - Marketing Outlook

The purpose of the project was to find target customers for our travel recommender and provide real life business solutions on how to add value based on what we know.

There were 4 major user groups and based on the findings we were able to conclude the target group for each consumer. 

# SOLUTIONS

personalized Travel, Travel loyalty program, and creating an integration through booking platforms.

# VALUE CREATED:

1- User Generated Content(Blog, or social media community)
2- Partnerships with business that are apart of the interest groups of user groups

Happy traveling and exploring Europe with our unsupervised travel recommendations and personalized Twilio travel recommender system! If you have any questions or feedback, please don't hesitate to send message.
