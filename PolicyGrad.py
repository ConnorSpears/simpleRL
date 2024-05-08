import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym
import numpy as np

# Create a simple policy model
class Policy(tf.keras.Model):
    def __init__(self, action_size):
        super().__init__()
        self.dense1 = Dense(24, activation='relu')
        self.dense2 = Dense(24, activation='relu')
        self.output_layer = Dense(action_size, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# Function to run an episode and return trajectories
def run_episode(environment, model):
    state = environment.reset()
    states, actions, rewards = [], [], []
    done = False
    while not done:
        state = np.expand_dims(state, axis=0)
        action_prob = model(state)
        action = np.random.choice(environment.action_space.n, p=action_prob.numpy()[0])
        next_state, reward, done, _ = environment.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
    
    return states, actions, rewards

# Function to calculate discounted rewards
def discount_rewards(rewards, gamma=0.99):
    discounted = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted[t] = running_add
    return discounted

# Training function
def train(policy_model, environment, optimizer, episodes=1000):
    for episode in range(episodes):
        states, actions, rewards = run_episode(environment, policy_model)
        total_reward = sum(rewards)
        discounted_rewards = discount_rewards(rewards)

        with tf.GradientTape() as tape:
            state_tensor = tf.concat(states, axis=0)
            action_probs = policy_model(state_tensor)
            indices = tf.range(action_probs.shape[0]) * action_probs.shape[1] + actions
            action_probs = tf.gather(tf.reshape(action_probs, [-1]), indices)
            loss = -tf.reduce_mean(tf.math.log(action_probs) * discounted_rewards)

        gradients = tape.gradient(loss, policy_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, policy_model.trainable_variables))


        print("out")
        
        if episode % 100 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}')

# Main function to create environment and train model
# env = gym.make('CartPole-v1')
# action_size = env.action_space.n

# policy_model = Policy(action_size)
# optimizer = Adam(learning_rate=0.01)

# #train(policy_model, env, optimizer)

