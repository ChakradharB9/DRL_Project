import numpy as np
import pandas as pd
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load datasets
ratings_df = pd.read_csv('rating.csv')
tags_df = pd.read_csv('tag.csv')
genome_scores_df = pd.read_csv('genome_scores.csv')
genome_tags_df = pd.read_csv('genome_tags.csv')
links_df = pd.read_csv('link.csv')
movies_df = pd.read_csv('movie.csv')

# Define state representation
state_representation = pd.merge(ratings_df, movies_df[['movieId', 'title']], on='movieId')

# Define action space
action_space = movies_df['movieId'].tolist()

# Define reward function
def get_reward(user_id, movie_id):
    rating = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['movieId'] == movie_id)]['rating'].values
    return rating[0] if len(rating) > 0 else 0

# Define Deep Q-Learning agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
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
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))  # Update here
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.sample(action_space, 1)[0]
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

# Initialize DQNAgent
state_size = 1  # State size (number of features)
action_size = len(action_space)  # Action size (number of possible actions)
agent = DQNAgent(state_size, action_size)

# Training parameters
batch_size = 32
EPISODES = 1000

# Deep Q-Learning training
for e in range(EPISODES):
    state = np.array([random.choice(action_space)]).reshape(1, state_size)
    for time in range(100):
        action = agent.act(state)
        reward = get_reward(1, action)  # Placeholder user_id=1
        next_state = np.array([random.choice(action_space)]).reshape(1, state_size)
        done = False  # Reset done flag at the beginning of each episode
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if time == 99:  # If max time steps reached, set done flag to True
            done = True
        if done:
            print("Episode:", e + 1, "/", EPISODES)  # Print episode number
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    # Terminate loop if the current episode count reaches the total number of episodes
    if e + 1 >= EPISODES:
        break

# Function to make movie recommendations using the trained model
def make_recommendations(user_id):
    state = np.array([random.choice(action_space)]).reshape(1, state_size)
    recommended_movie = agent.act(state)
    return recommended_movie

# Example: Make recommendations for user with ID 1
user_id = 1
recommended_movie = make_recommendations(user_id)
print("Recommended Movie ID:", recommended_movie)
