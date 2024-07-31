import time
from src.Simulator import TiltrotorTransitionSimulator
from stable_baselines3 import SAC

class Tester:
    def __init__(self, modelName, simulator_name="Simulator", waitingTime=1):
        self.env = TiltrotorTransitionSimulator()
        self.loadModel(modelName=modelName)
        self.waitingTime = waitingTime
        self.flightData = self.initData()
        self.actionData = self.initActionData()

    def loadModel(self, modelName):
        self.model = SAC.load(f'./src/model/{modelName}')

    def setWatingTime(self, second):
        self.waitingTime = second

    def runTest(self, maximum_timestep=100):
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
            "frontRPM": [],
            "rearRPM": [],
            "U": [], 
            "tilt_deg": [],
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
        self.flightData["frontRPM"].append(data["frontRPM"])
        self.flightData["rearRPM"].append(data["rearRPM"])
        self.flightData["U"].append(data["U"])
        self.flightData["tilt_deg"].append(data["tilt_deg"])
        self.flightData["Lift"].append(data["Lift"])
        self.flightData["L/W"].append(data["L/W"])
        self.flightData["(L+T)/W"].append(data["(L+T)/W"])

    def addActionData(self, action):
        print(f'action: {action}')
        self.actionData["action_0"].append(action[0])
        self.actionData["action_1"].append(action[1])
        self.actionData["action_2"].append(action[2])
        self.actionData["action_3"].append(action[3])

    