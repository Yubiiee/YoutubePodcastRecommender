import pandas as pd
import requests
import nltk
import re
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

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

    def extract_keywords(self, user_input):
        """
        Extract keywords from the user input for a more flexible search.
        """
        stemmer = PorterStemmer()
        user_input = re.sub(r'[^\w\s]', '', user_input.lower())
        tokens = [word for word in user_input.split() if word not in stopwords.words('english')]
        
        # Stem the tokens
        keywords = [stemmer.stem(word) for word in tokens]
        return keywords

    def get_recommendations(self, keywords, num_recommendations=5):
        """Get podcast recommendations based on keywords if an exact match is not found."""
        if self.df.empty or self.cosine_sim is None:
            print("Data is not ready. Make sure to fetch data and build the similarity matrix first.")
            return pd.DataFrame()

        # Filter descriptions containing any of the keywords
        keyword_pattern = "|".join(keywords)  # Create a pattern from the keywords
        filtered_df = self.df[self.df['Processed_Text'].str.contains(keyword_pattern, na=False, case=False)]
        
        if not filtered_df.empty:
            # If matching podcasts found, return them as recommendations
            return filtered_df[['title', 'description']].head(num_recommendations)
        else:
            print("No podcasts found based on the keywords.")
            return pd.DataFrame()

    def recommend_podcasts(self, user_input, num_recommendations=5):
        """
        Recommend podcasts based on keywords from user input, or exact title if available.
        """
        if not self.fetch_podcast_data():
            return pd.DataFrame()
        self.build_similarity_matrix()
        
        # Extract keywords from the user input
        keywords = self.extract_keywords(user_input)
        
        # First, try to find an exact match
        exact_match = self.df[self.df['title'].str.contains(user_input, case=False)]
        if not exact_match.empty:
            print("Exact match found:")
            return exact_match[['title', 'description']].head(num_recommendations)
        
        # If no exact match, use keywords to find similar podcasts
        print("No exact match found, searching based on keywords...")
        return self.get_recommendations(keywords, num_recommendations)

# Example usage with user input
if __name__ == "__main__":
    api_key = "AIzaSyD8ew2kHDfO-p3-PweFKj44W4fd7hdggj4"  
    youtube_recommender = YouTubePodcastRecommender(api_key)
    
    # Prompt the user to enter a podcast title or topic
    user_input = input("Enter the podcast title or topic you're interested in: ")
    
    # Optional: prompt for the number of recommendations
    try:
        num_recommendations = int(input("Enter the number of recommendations you'd like: "))
    except ValueError:
        num_recommendations = 10  

    # Get recommendations based on user input
    recommendations = youtube_recommender.recommend_podcasts(user_input, num_recommendations)
    print(recommendations)

