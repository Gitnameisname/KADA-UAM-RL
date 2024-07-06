import torch as th
from src.Simulator import TiltrotorTransitionSimulator
from stable_baselines3 import SAC
from datetime import datetime
import os

class Trainer:
    def __init__(self, device):
        self.filename = self.get_filename()
        self.device = device
        self.env = TiltrotorTransitionSimulator()
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[256, 256, 256], qf=[128, 128]))
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
        self.model = SAC.load(f'./src/model/{modelName}', env=self.env, device=self.device)

    def startTraining(self, maximum_eps=1000, maximum_timestep=10000):
        best_reward = 0
        result_dir = 'src/results'
        model_dir = 'src/model'
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        with open(f'{result_dir}/{self.filename}.txt', 'w') as file:
            for eps in range(maximum_eps):
                obs = self.env.reset()  # 환경 초기화
                self.model.learn(total_timesteps=maximum_timestep, log_interval=100)

                for k in range(maximum_timestep):
                    action, _states = self.model.predict(obs, deterministic=True)  # 액션 예측
                    obs, reward, dones, infos = self.env.step(action)  # 환경 스텝 실행
                    self.env.render()
                    if dones:  # 에피소드 종료 시 처리
                        log = (f"Episode: {eps}\nEpisode finished at timestep: {k}\n reward: {reward}\n")
                        log += "========================================\n"
                        log += "Reward details\n"
                        log += f"Tilt       : {infos['reward_detail'][0]:.2f}\t| {infos['value_detail'][0]:.2}\n"
                        log += f"Pitch      : {infos['reward_detail'][1]:.2f}\t| {infos['value_detail'][1]:.2}\n"
                        log += f"Time       : {infos['reward_detail'][2]:.2f}\t| {infos['value_detail'][2]:.2}\n"
                        log += f"V_cruise   : {infos['reward_detail'][3]:.2f}\t| {infos['value_detail'][3]:.2}\n"
                        log += f"Altitude   : {infos['reward_detail'][4]:.2f}\t| {infos['value_detail'][4]:.2}\n"
                        log += f"Rotor RPM  : {infos['reward_detail'][5]:.2f}\t| front - {infos['value_detail'][5][0]:.2}\t| rear - {infos['value_detail'][5][1]:.2}\n"
                        log += f"Distance   : {infos['reward_detail'][6]:.2f}\t| {infos['value_detail'][6]:.2}\n"
                        log += f"G-Force    : {infos['reward_detail'][7]:.2f}\t| {infos['value_detail'][7]:.2}\n"
                        log += f"Tilt Delta : {infos['reward_detail'][8]:.2f}\t| {infos['value_detail'][8]:.2}\n"
                        log += "========================================\n"
                        file.write(f'{log}\n')

                # 이번 에피소드의 총 보상이 최상의 보상보다 좋다면 모델 저장
                if reward > best_reward:
                    best_reward = reward
                    try:
                        self.model.save(f'{model_dir}/{self.filename}')
                    except Exception as e:
                        print(f"Error saving model: {e}")

            

