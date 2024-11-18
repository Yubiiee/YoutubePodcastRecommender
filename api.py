import pandas as pd
import requests
import nltk
import re
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

class PodcastRecommender:
    def __init__(self, client_id, client_secret):
        """
        Initialize the PodcastRecommender with Spotify API credentials.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = self.get_access_token()
        self.df = pd.DataFrame()
        self.cosine_sim = None

    def get_access_token(self):
        """
        Authenticate with Spotify and obtain an access token.
        """
        auth_url = "https://accounts.spotify.com/api/token"
        auth_data = {"grant_type": "client_credentials"}
        auth_headers = {
            "Authorization": "Basic " + base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        }
        
        response = requests.post(auth_url, data=auth_data, headers=auth_headers)
        response_data = response.json()
        return response_data.get("access_token")

    def fetch_podcast_data(self, query="technology", market="US", limit=50):
        """
        Fetch podcast data from the Spotify API based on a search query.
        """
        api_url = "https://api.spotify.com/v1/search"
        headers = {"Authorization": f"Bearer {self.token}"}
        params = {"q": query, "type": "show", "market": market, "limit": limit}

        response = requests.get(api_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            shows = data.get("shows", {}).get("items", [])
            self.df = pd.DataFrame(shows)  # Create a DataFrame with the results
            self.df['Processed_Text'] = self.df['description'].apply(self.preprocess_text)
            return True
        else:
            print("Failed to fetch data:", response.status_code, response.text)
            return False
        

    def preprocess_text(self, text):
        """
        Preprocess the text by removing punctuation, lowering case, removing stop words,
        and expanding synonyms.
        """
        stemmer = PorterStemmer()
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = [word for word in text.split() if word not in stopwords.words('english')]
        
        expanded_tokens = []
        for word in tokens:
            synonyms = {lemma.name() for syn in wn.synsets(word) for lemma in syn.lemmas()}
            expanded_tokens.extend([word, *synonyms])
        
        return ' '.join(stemmer.stem(word) for word in expanded_tokens)

    def build_similarity_matrix(self):
        """
        Build a cosine similarity matrix from the processed text data.
        """
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['Processed_Text'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def get_recommendations(self, podcast_title, num_recommendations=5):
        """Get podcast recommendations based on the cosine similarity of the descriptions."""
        if self.df.empty or self.cosine_sim is None:
            print("Data is not ready. Make sure to fetch data and build the similarity matrix first.")
            return pd.DataFrame()
        print("Available podcasts:", self.df['name'].tolist())

        try:
            podcast_index = self.df[self.df['name'] == podcast_title].index[0]  # Spotify returns title as 'name'
            print(f"Found podcast '{podcast_title}' at index {podcast_index}")  # Debugging line
            sim_scores = sorted(enumerate(self.cosine_sim[podcast_index]), key=lambda x: x[1], reverse=True)
            podcast_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]
            return self.df.iloc[podcast_indices][['name', 'description']]
        except IndexError:
            print(f"Podcast title '{podcast_title}' not found.")
            return pd.DataFrame()

    def recommend_podcasts(self, podcast_title, num_recommendations=5):
        """
        Recommend podcasts by fetching data, building similarity matrix, and getting recommendations.
        """
        if not self.fetch_podcast_data():
            return pd.DataFrame()
        self.build_similarity_matrix()
        return self.get_recommendations(podcast_title, num_recommendations)
    


if __name__ == "__main__":
    client_id = "f4becc97fd204cc5b60046e288770bae"
    client_secret = "e57e2998eb254b968e41dc3e8e46d233"
    
    podcast_recommender = PodcastRecommender(client_id, client_secret)
    podcast_title = "The Vergecast"  
    recommendations = podcast_recommender.recommend_podcasts(podcast_title, num_recommendations=5)
    print(recommendations)