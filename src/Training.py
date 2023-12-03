import torch as th
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
        self.model = SAC.load(f'./model/{modelName}', env = self.env, device=self.device)

    def startTraining(self, maximum_eps=1000, maximum_timestep=10000):
        best_reward = 0

        with open(f'./results/{self.filename}.txt', 'w') as file:
            for eps in range(maximum_eps):
                obs = self.env.reset()
                self.model.learn(total_timesteps=maximum_timestep, log_interval=100)
                for k in range(maximum_timestep):
                    action, _states = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.env.step(action)
                    self.env.render()
                    if done:
                        log = f"episode: {eps}\nepisode was finished at timestep: {k}\nreward: {reward}\n========================================\n" + \
                        f"reward details\n" + \
                        f"reward_1: {info['reward_detail'][0]}\n" + \
                        f"reward_2: {info['reward_detail'][1]}\n" + \
                        f"reward_3: {info['reward_detail'][2]}\n" + \
                        f"reward_4: {info['reward_detail'][3]}\n" + \
                        f"reward_5: {info['reward_detail'][4]}\n" + \
                        f"reward_6: {info['reward_detail'][5]}\n" + \
                        f"reward_7: {info['reward_detail'][6]}\n" + \
                        "========================================\n"
                        print(log)
                        file.write(f'{log}\n')

                        break
                # 이번 학습 결과가 기존 학습 결과보다 좋다면 저장
                if reward > best_reward:
                    best_reward = reward
                    self.model.save(f'./model/{self.filename}')