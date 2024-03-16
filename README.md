# Recommendation-system---Book-problem
Problem statement.  Build a recommender system by using cosine simillarties score.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset from the CSV file
data = pd.read_csv('book.csv', encoding='latin1')

# Drop duplicate entries
data = data.drop_duplicates()

# Visualizations
# Histogram of book ratings
plt.figure(figsize=(10, 6))
sns.histplot(data['Book.Rating'], bins=10, kde=True)
plt.title('Histogram of Book Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Scatterplot of book ratings for a specific user
user_id = 276747  # Example user ID
user_ratings = data[data['User.ID'] == user_id]
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Book.Title', y='Book.Rating', data=user_ratings)
plt.title(f'Book Ratings for User ID {user_id}')
plt.xticks(rotation=90)
plt.xlabel('Book Title')
plt.ylabel('Rating')
plt.show()

# Pairplot to explore relationships between different books
plt.figure(figsize=(12, 8))
sns.pairplot(data.sample(1000), kind='scatter', diag_kind='kde')
plt.title('Pairplot of Book Ratings')
plt.show()

# Compute the cosine similarity matrix
user_item_matrix = data.pivot_table(index='User.ID', columns='Book.Title', values='Book.Rating', aggfunc='mean').fillna(0)
cosine_sim_matrix = cosine_similarity(user_item_matrix, user_item_matrix)

def get_similar_books(book_title, top_n=5):
    # Get the index of the given book title
    book_index = data[data['Book.Title'] == book_title].index[0]

    # Get the cosine similarity scores for the book
    book_sim_scores = list(enumerate(cosine_sim_matrix[book_index]))

    # Sort the scores in descending order
    book_sim_scores = sorted(book_sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top N similar books
    top_similar_books = book_sim_scores[1:top_n+1]

    # Get the indices and titles of the top similar books
    similar_book_indices = [book[0] for book in top_similar_books]
    similar_book_titles = data.iloc[similar_book_indices]['Book.Title'].values

    return similar_book_titles

book_title = 'Classical Mythology'
similar_books = get_similar_books(book_title)
print(f"Books similar to '{book_title}':")
for book in similar_books:
    print(book)
