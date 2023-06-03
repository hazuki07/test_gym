import gym
import numpy as np
import env.environment as environment

class QLearningAgent:
    def __init__(self, env, epsilon=1.0, lr=0.5, gamma=0.95):
        self.env = env
        self.observation_space = self.env.observation_space.high[0].astype(int)  # Change here
        self.action_space = self.env.action_space.n
        self.q_table = np.zeros((self.observation_space, self.action_space))
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma

    def step(self, state):
        # ランダムな数字がepsilonより小さい場合、探索（explore）を行い、そうでない場合は活用（exploit）します
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])

        # Q値の更新
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

        # epsilonの減衰
        if self.epsilon > 0.01:
            self.epsilon -= 0.01
            
    def play(self, state):
        """
        学習後のプレイ用メソッド。
        探索（ランダムな行動選択）を行わず、
        すべての行動が活用（最適な行動の選択）に基づいています。
        """
        return np.argmax(self.q_table[state])



env = gym.make('GolfEnv-v0', new_step_api=True)
agent = QLearningAgent(env)

episodes = 10

for _ in range(episodes):
    state = env.reset().astype(int)
    done = False

    while not done:
        action = agent.step(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.update(state, action, reward, next_state.astype(int))
        # env.render()
        state = next_state.astype(int)
        

episodes = 10
for _ in range(episodes):
    state = env.reset().astype(int)
    done = False

    while not done:
        action = agent.play(state)
        next_state, reward, done, _, _ = env.step(action)
        env.render()
        state = next_state.astype(int)

env.close()