import numpy as np
import pandas as pd
import random
from math import sqrt, log

# Load merged dataset
movies_df = pd.read_csv('movie.csv')

# Function to get user's preferred genre
def get_user_genre():
    while True:
        genre = input("Enter your preferred movie genre: ")
        if genre.lower() in movies_df['genres'].str.lower().unique():
            return genre.lower()
        else:
            print("Invalid genre. Please choose from the available genres.")

# Function to ask if the user has seen the recommended movie
def ask_seen_movie(recommended_movie):
    while True:
        seen_movie = input(f"Have you seen '{recommended_movie}'? (yes/no): ")
        if seen_movie.lower() == 'yes' or seen_movie.lower() == 'no':
            return seen_movie.lower()
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

# Function to ask if the user wants another recommendation from the same genre or a different genre
def ask_another_recommendation():
    while True:
        another_recommendation = input("Do you want another recommendation from the same genre? (yes/no): ")
        if another_recommendation.lower() == 'yes' or another_recommendation.lower() == 'no':
            return another_recommendation.lower()
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

# Node class for the MCTS algorithm
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0

    def select_child(self):
        if not self.children:
            return None
        max_child = None
        max_score = float('-inf')
        for child in self.children:
            if child.visits == 0:
                score = float('inf')  # Set score to infinity if child has not been visited
            else:
                score = child.reward / child.visits + sqrt(2 * log(self.visits) / child.visits)
            if score > max_score:
                max_score = score
                max_child = child
        return max_child

    def expand(self, action_space):
        for action in action_space:
            self.children.append(Node(action, parent=self))

    def update(self, reward):
        self.visits += 1
        self.reward += reward
        if self.parent:
            self.parent.update(reward)

# Monte Carlo Tree Search (MCTS) algorithm
class MCTS:
    def __init__(self, root_state, action_space):
        self.root = Node(root_state)
        self.action_space = action_space

    def search(self, num_simulations):
        for _ in range(num_simulations):
            node = self.root
            while node.children:
                node = node.select_child()
            if node.visits > 0:
                node.expand(self.action_space)
                node = random.choice(node.children)
            reward = 1  # Placeholder reward for demonstration
            while node:
                node.update(reward)
                node = node.parent

# Main loop for recommending movies
while True:
    user_genre = get_user_genre()
    movies_in_genre = movies_df[movies_df['genres'].str.lower().str.contains(user_genre)]
    
    if movies_in_genre.empty:
        print("Sorry, there are no movies available in the selected genre. Please choose another genre.")
        continue

    while True:
        root_state = movies_in_genre['title'].sample().iloc[0]
        mcts = MCTS(root_state, movies_in_genre['title'].tolist())
        num_simulations = 10000
        mcts.search(num_simulations)
        recommended_movie = max(mcts.root.children, key=lambda child: child.reward / child.visits).state

        print("Recommended Movie:", recommended_movie)
        seen_movie = ask_seen_movie(recommended_movie)

        if seen_movie == 'no':
            another_recommendation = ask_another_recommendation()
            if another_recommendation == 'no':
                break
        else:
            print("Oh alright!")
            another_recommendation = ask_another_recommendation()
            if another_recommendation == 'no':
                break
            else:
                user_genre = get_user_genre()
                movies_in_genre = movies_df[movies_df['genres'].str.lower().str.contains(user_genre)]
                if movies_in_genre.empty:
                    print("Sorry, there are no movies available in the selected genre. Please choose another genre.")
                    break
                continue

    if input("Do you want to continue? (yes/no): ").lower() != 'yes':
        break
