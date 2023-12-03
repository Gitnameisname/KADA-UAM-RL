import time
from src.Simulator import TiltrotorTransitionSimulator
from stable_baselines3 import SAC

class Tester:
    def __init__(self, modelName):
        self.env = TiltrotorTransitionSimulator()
        self.model = self.loadModel(modelName=modelName)
        self.waitingTime = 5 # seconds
    
    def loadModel(self, modelName):
        self.model = SAC.load(f'./model/{modelName}')

    def setWatingTime(self, second):
        self.waitingTime = second

    def runTest(self):
        time.sleep(self.waitingTime)
        obs = self.env.reset()

        for k in range(3000):
            action, _states = self.model.predict(obs, deterministic=True)

            obs, reward, done, info = self.env.step(action)
            #print(action)
            self.env.render()
            if done:
                obs = self.env.reset()

    