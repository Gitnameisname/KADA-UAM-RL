import time
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecExtractDictObs
from stable_baselines3 import SAC

class Tester:
    def __init__(self, modelName, simulator_name="Simulator", num_envs=1):
        self.simulator_name = simulator_name
        self.num_envs = num_envs
        self.env = self.create_vec_envs()
        self.loadModel(modelName=modelName)
        self.waitingTime = 5 # seconds
        self.flightData = self.initData()
        self.actionData = self.initActionData()
    
    def create_vec_envs(self):
        def make_env():
            def _init():
                if self.simulator_name == "Simulator":
                    from src.Simulator import TiltrotorTransitionSimulator
                    return TiltrotorTransitionSimulator()
                elif self.simulator_name == "Simulator_7act":
                    from src.Simulator_7act import TiltrotorTransitionSimulator
                    return TiltrotorTransitionSimulator()
            return _init

        envs = [make_env() for _ in range(self.num_envs)]
        env = DummyVecEnv(envs)
        env = VecMonitor(env)
        # env = VecExtractDictObs(env, "observation")
        return env


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
            # self.addFlightData(info['data'])
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
        print(f'action: {action}')
        self.actionData["action_0"].append(action[0][0])
        self.actionData["action_1"].append(action[0][1])
        self.actionData["action_2"].append(action[0][2])
        self.actionData["action_3"].append(action[0][3])

    