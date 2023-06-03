import numpy as np
import gym
from gym import spaces

class GolfEnv(gym.Env):
    # 常に左に進むことを学習する環境
    LEFT = 0
    RIGHT = 1

    def __init__(self, grid_size=5):
        super(GolfEnv, self).__init__()
        self.grid_size = grid_size   
        self.agent_pos = grid_size - 1 # エージェントの状態を定義
        n_actions = 2                  # 行動は「0:LEFT, 1:RIGHT」の二通り
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0, high=self.grid_size,shape=(1,), dtype=np.float32)

    def reset(self, seed=None, return_info=False, options=None):
        # 必要に応じてここでランダムシードを設定します
        if seed is not None:
            np.random.seed(seed)

        # 既存のresetロジックをここに書く
        self.agent_pos = self.grid_size - 1

        return np.array([self.agent_pos]).astype(np.float32)


    def step(self, action):
        if action == self.LEFT:    # 0
            self.agent_pos -= 1
        elif action == self.RIGHT: # 1
            self.agent_pos += 1
        else:
            raise ValueError(f'Received invalid action={action}' + f' is not part of the action space.')

        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)
        done = bool(self.agent_pos == 0)         # 終了判定左端(0)に着いたら終了とする
        reward = 1 if self.agent_pos == 0 else 0
        info = {}

        return np.array([self.agent_pos]).astype(np.float32), reward, done, done, info


    def render(self, mode='console'):
        # 環境をコンソール上に描画するメソッド

        if mode != 'console':
            raise NotImplementedError()
        print('.' * self.agent_pos, end='')
        print('x', end='')
        print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass