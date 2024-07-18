import torch as th
import math
from src.Simulator import TiltrotorTransitionSimulator
from stable_baselines3 import SAC
from datetime import datetime

class Trainer:
    def __init__(self, device):
        self.filename = self.get_filename()

        # Parallel environments
        self.env = TiltrotorTransitionSimulator()
        self.device = device
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[256,256,256], qf=[128,128]))
        self.model = SAC("MlpPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1, device=device)

    def get_filename(self):
        year = str(datetime.today().year)
        month = str(datetime.today().month).zfill(2)
        day = str(datetime.today().day).zfill(2)
        hour = str(datetime.today().hour).zfill(2)
        minute = str(datetime.today().minute).zfill(2)
        second = str(datetime.today().second).zfill(2)

        filename = f'train_result_{year}-{month}-{day}_{hour}{minute}{second}'
        return filename
    
    def loadModel(self, modelName):
        self.model = SAC.load(f'./src/model/{modelName}', env = self.env, device=self.device)

    def startTraining(self, maximum_eps=1000, maximum_timestep=10000):
        best_reward = 0

        with open(f'./src/results/{self.filename}.txt', 'w') as file:
            for eps in range(maximum_eps):
                obs = self.env.reset()
                self.model.learn(total_timesteps=maximum_timestep, log_interval=100)
                for k in range(maximum_timestep):
                    action, _states = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.env.step(action)
                    self.env.render()
                    if done:
                        log = (f"Episode: {eps}\nEpisode finished at timestep: {k}\nTotal reward: {reward:.2f}\n"
                            "========================================\n"
                            "Reward details\n"
                            f"Tilt       : {info['reward_detail'][0]:.2f}\t| {info['value_detail'][0]:.2}\n"
                            f"Pitch      : {info['reward_detail'][1]:.2f}\t| {info['value_detail'][1]:.2}\n"
                            f"Time       : {info['reward_detail'][2]:.2f}\t| {info['value_detail'][2]:.2}\n"
                            f"V_cruise   : {info['reward_detail'][3]:.2f}\t| {info['value_detail'][3]:.2}\n"
                            f"Altitude   : {info['reward_detail'][4]:.2f}\t| {info['value_detail'][4]:.2}\n"
                            f"Rotor RPM  : {info['reward_detail'][5]:.2f}\t| front - {info['value_detail'][5][0]:.2}\t| rear - {info['value_detail'][5][1]:.2}\n"
                            f"Distance   : {info['reward_detail'][6]:.2f}\t| {info['value_detail'][6]:.2}\n"
                            f"G-Force    : {info['reward_detail'][7]:.2f}\t| {info['value_detail'][7]:.2}\n"
                            f"Tilt Delta : {info['reward_detail'][8]:.2f}\t| {info['value_detail'][8]:.2}\n"
                            "========================================\n")
                        print(log)
                        file.write(f'{log}\n')

                        break
                # 이번 학습 결과가 기존 학습 결과보다 좋다면 저장
                if reward > best_reward:
                    best_reward = reward
                    self.model.save(f'./src/model/{self.filename}')