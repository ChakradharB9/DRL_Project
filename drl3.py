import numpy as np
import pandas as pd
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load datasets
ratings_df = pd.read_csv('rating.csv')
movies_df = pd.read_csv('movie.csv')
tags_df = pd.read_csv('tag.csv')
genome_scores_df = pd.read_csv('genome_scores.csv')
genome_tags_df = pd.read_csv('genome_tags.csv')
links_df = pd.read_csv('link.csv')

# Merge datasets
movies_tags_df = pd.merge(movies_df, tags_df, on='movieId')
movies_ratings_df = pd.merge(movies_df, ratings_df, on='movieId')

# Define action space (movie titles)
action_space = movies_df['title'].tolist()

# Define reward function
def get_reward(user_preferences, movie_title):
    movie_rating = movies_ratings_df[movies_ratings_df['title'] == movie_title]['rating'].mean()
    if pd.isna(movie_rating):
        movie_rating = 0  # Assign 0 if movie has no ratings
    movie_tags = movies_tags_df[movies_tags_df['title'] == movie_title]['tag'].tolist()
    match_count = sum(1 for tag in user_preferences if tag in movie_tags)
    return (movie_rating + match_count) / (len(user_preferences) + 1)  # Normalize to [0, 1]

# Define Deep Q-Learning agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)  # Limit memory size
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(action_space)
        act_values = self.model.predict(state)
        return action_space[np.argmax(act_values[0])]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action_space.index(action)] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Train for a limited number of episodes
def train_episodes(agent, episodes, user_preferences):
    state_size = len(action_space)
    for e in range(episodes):
        print("Episode:", e + 1, "/", episodes)  # Print episode number
        state_index = random.randrange(len(action_space))
        state = np.zeros((1, len(action_space)))
        state[0, state_index] = 1
        for time in range(50):  # Reduce episode length
            action = agent.act(state)
            reward = get_reward(user_preferences, action)  # Reward based on user preferences
            next_state_index = random.randrange(len(action_space))
            next_state = np.zeros((1, len(action_space)))
            next_state[0, next_state_index] = 1
            done = False
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if time == 49:
                done = True
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e + 1 >= episodes:
            break  # Terminate loop after completing the specified number of episodes

if __name__ == '__main__':
    # Get user preferences
    user_preferences = input("Enter your movie preferences (comma-separated): ").split(',')

    # Initialize DQNAgent
    state_size = len(action_space)
    action_size = len(action_space)
    agent = DQNAgent(state_size, action_size)

    # Training parameters
    batch_size = 32
    episodes = 1000  # Set number of episodes to 5

    # Train episodes
    train_episodes(agent, episodes, user_preferences)

    # Function to make movie recommendations using the trained model
    def make_recommendations(user_preferences):
        state_index = random.randrange(len(action_space))
        state = np.zeros((1, len(action_space)))
        state[0, state_index] = 1
        recommended_movie = agent.act(state)
        return recommended_movie

    # Initially recommended movie
    recommended_movie = make_recommendations(user_preferences)
    print("Recommended Movie:", recommended_movie)

    # Loop to recommend movies
    while True:
        seen_movie = input("Have you seen this movie? (yes/no): ")
        if seen_movie.lower() == 'no':
            recommended_movie = make_recommendations(user_preferences)
            print("Recommended Movie:", recommended_movie)
        elif seen_movie.lower() == 'yes':
            agree = input("Do you agree with this recommendation? (yes/no): ")
            if agree.lower() == 'yes':
                print("Enjoy your movie!")
                break
            elif agree.lower() == 'no':
                recommended_movie = make_recommendations(user_preferences)
                print("Recommended Movie:", recommended_movie)
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
