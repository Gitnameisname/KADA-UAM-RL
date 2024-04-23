import time
from stable_baselines3 import SAC

class Tester:
    def __init__(self, modelName, simulator_name="Simulator"):
        if simulator_name == "Simulator":
            from src.Simulator import TiltrotorTransitionSimulator
        elif simulator_name == "Simulator_7act":
            from src.Simulator_7act import TiltrotorTransitionSimulator
        self.env = TiltrotorTransitionSimulator()
        self.loadModel(modelName=modelName)
        self.waitingTime = 5 # seconds
        self.flightData = self.initData()
        self.actionData = self.initActionData()
    
    def loadModel(self, modelName):
        self.model = SAC.load(f'./src/model/{modelName}')

    def setWatingTime(self, second):
        self.waitingTime = second

    def runTest(self, maximum_timestep=3000):
        time.sleep(self.waitingTime)
        obs = self.env.reset()

        for k in range(maximum_timestep):
            action, _states = self.model.predict(obs, deterministic=True)

            self.addActionData(action)

            obs, reward, done, info = self.env.step(action)
            self.addFlightData(info['data'])
            self.env.render()
            if done:
                obs = self.env.reset()

    def initData(self):
        data = {
            "time": [],
            "CL": [],
            "CD": [],
            "X": [],
            "Z": [],
            "theta": [],
            "f_rpm": [],
            "r_rpm": [],
            "U": [], 
            "tilt": [],
            "Lift": [],
            "L/W": [],
            "(L+T)/W": []
        }
        return data
    
    def initActionData(self):
        actionData = {
            "action_0": [],
            "action_1": [],
            "action_2": [],
            "action_3": []
        }
        return actionData

    def addFlightData(self, data):
        self.flightData["time"].append(data["time"])
        self.flightData["CL"].append(data["CL"])
        self.flightData["CD"].append(data["CD"])
        self.flightData["X"].append(data["X"])
        self.flightData["Z"].append(data["Z"])
        self.flightData["theta"].append(data["theta"])
        self.flightData["f_rpm"].append(data["f_rpm"])
        self.flightData["r_rpm"].append(data["r_rpm"])
        self.flightData["U"].append(data["U"])
        self.flightData["tilt"].append(data["tilt"])
        self.flightData["Lift"].append(data["Lift"])
        self.flightData["L/W"].append(data["L/W"])
        self.flightData["(L+T)/W"].append(data["(L+T)/W"])

    def addActionData(self, action):
        self.actionData["action_0"].append(action[0])
        self.actionData["action_1"].append(action[1])
        self.actionData["action_2"].append(action[2])
        self.actionData["action_3"].append(action[3])

    