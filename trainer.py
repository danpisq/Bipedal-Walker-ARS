import gym
import numpy as np



class Model:
    """
    Class representing the policy
    """
    def __init__(self, input_size, output_size):
        self.weights = np.zeros((input_size, output_size))
        
    # Returns predicted action which is dot product of weights and input
    # if deltas are given output is dot product of input and weights updated by deltas
    def predict(self, inp, deltas=None):
        w = self.weights
        if deltas:
            w += deltas
        output = np.dot(inp, w)
        return output
    
    # returns model weights
    def get_weights(self):
        return self.weights
        
    # sets model weights
    def set_weights(self, weights):
        self.weights = weights
        
class Normalizer:
    """
    Normalzies the input to be between 0 and 1
    """
    def __init__(self, input_size):
        self.n = np.zeros( input_size)
        self.mean = np.zeros( input_size)
        self.mean_diff = np.zeros( input_size)
        self.std = np.zeros(input_size)

    # given new data it updates parametest of normalizer
    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    # normalizes the input
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

class Agent:
    """
    Class used to train model with ARS
    """

    GAME = 'BipedalWalker-v2'

    def __init__(self):
        self.env = gym.make(self.GAME)
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.shape[0]
        self.model = Model(self.input_size, self.output_size)
        self.normalizer = Normalizer(self.input_size)
        self.noise_rate = 0.06
        self.alpha = 0.09
        self.population = 16
        np.random.seed(1)
    
    # plays an episode of a game
    def play(self, deltas=None, render=False):
        total_reward = 0
        observation = self.env.reset()
        n=0   
        while n < 2000:
            if render:
                self.env.render()
            self.normalizer.observe(observation)
            observation = self.normalizer.normalize(observation)
            action = self.model.predict(observation, deltas)
            observation, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            total_reward += reward
            n+=1
            if done:
                break
        return total_reward
        
    # first training method uses only one deltas each step
    # ( less effective )
    def train(self, n_steps):
        for step in range(n_steps):
            deltas = self.noise_rate * np.random.random((self.input_size, self.output_size))
            old_weights = self.model.get_weights()
            r_p = self.play(deltas=deltas,render=False)
            r_n = self.play(deltas,deltas,render=False)
            
            new_weights = old_weights + self.alpha * (r_p - r_n) * deltas
            self.model.set_weights(new_weights)
            if step % 500 == 0:
                reward = self.play(render=True)
                print('Step: ', step, 'Reward+: ',reward )
    
    # second training method uses sel.population number of deltas each step
    # ( much more effective - faster training )
    def train_population(self, n_steps):
        for step in range(n_steps):
            #deltas = [self.noise_rate * np.random.randn(*self.model.weights.shape) for _ in range(self.population)]
            positive_rewards=[0] * self.population
            negative_rewards=[0] * self.population
            old_weights = self.model.get_weights()
            
            for k, d in enumerate(deltas):
                self.model.set_weights(old_weights + d)
                positive_rewards[k] = self.play(render=False)
                self.model.set_weights(old_weights - d)
                negative_rewards[k] = self.play(render=False)
                
            scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:self.population]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
            
            update = np.zeros(self.model.weights.shape)
            for r_p, r_n, delta in rollouts:
                update += (r_p - r_n) * delta
            new_weights = old_weights + self.alpha * update / (self.population)
            self.model.set_weights(new_weights)
            
            re = False
            if step % 100 == 0:
                re = True
            reward = self.play(render=re)
            print('Step: ', step, 'Reward: ',reward )
            
        
if __name__ == '__main__':
    agent = Agent()
    agent.train_population(500)
