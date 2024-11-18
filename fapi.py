import pandas as pd
import requests
import nltk
import re
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class YouTubePodcastRecommender:
    def __init__(self, api_key):
        """
        Initialize the YouTubePodcastRecommender with YouTube API credentials.
        """
        self.api_key = api_key
        self.df = pd.DataFrame()
        self.cosine_sim = None

    def fetch_podcast_data(self, query="technology podcast", max_results=50):
        """
        Fetch podcast data from the YouTube API based on a search query.
        """
        api_url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "key": self.api_key
        }

        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            
            # Extracting the title and description for each video result
            videos = [{"title": item["snippet"]["title"], "description": item["snippet"]["description"]} for item in items]
            self.df = pd.DataFrame(videos)  # Create a DataFrame with the results
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
        print("Available podcasts:", self.df['title'].tolist())

        try:
            podcast_index = self.df[self.df['title'] == podcast_title].index[0]
            print(f"Found podcast '{podcast_title}' at index {podcast_index}")  # Debugging line
            sim_scores = sorted(enumerate(self.cosine_sim[podcast_index]), key=lambda x: x[1], reverse=True)
            podcast_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]
            return self.df.iloc[podcast_indices][['title', 'description']]
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

# Example usage with user input
if __name__ == "__main__":
    api_key = "AIzaSyD8ew2kHDfO-p3-PweFKj44W4fd7hdggj4"  
    youtube_recommender = YouTubePodcastRecommender(api_key)
    
    # Prompt the user to enter a podcast title
    podcast_title = input("Enter the podcast title you're interested in: ")
    
    # Optional: prompt for the number of recommendations
    try:
        num_recommendations = int(input("Enter the number of recommendations you'd like: "))
    except ValueError:
        num_recommendations = 10  

    # Get recommendations based on user input
    recommendations = youtube_recommender.recommend_podcasts(podcast_title, num_recommendations)
    print(recommendations)
