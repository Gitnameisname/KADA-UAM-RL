import gym
import math
import pygame
import os
import numpy as np
from gym import spaces
from gym.utils import seeding

from src.loadDB import dataLoader

class TiltrotorTransitionSimulator(gym.Env):
    metadata = {'render.modes': ['human']}
    ################### __init__ ####################
    def __init__(self):
        self.set_render([1000,500])
        
        self.set_DB("aero.json")
        
        self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(11,), dtype=np.float32)
        
        # Continous Action
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)         
        self.set_init_state()
        self.seed()       
        
    def set_render(self, window_size):
        self.window_size = window_size
        img_path = "./src/image"

        self.background = pygame.image.load(os.path.join(f"{img_path}/Background.png"))

        # self.background = pygame.image.load(os.path.join("C:/Users/ds040/OneDrive - konkuk.ac.kr/Teams/RL/Transition-Train/images/Background.png"))
        self.vehicle = pygame.image.load(os.path.join(f"{img_path}/vehicle.png"))
        # self.vehicle = pygame.image.load(os.path.join("C:/Users/ds040/OneDrive - konkuk.ac.kr/Teams/RL/Transition-Train/images/vehicle.png"))
        vehicle_size = self.vehicle.get_rect().size
        self.vehicle_width = vehicle_size[0]
        self.vehicle_height = vehicle_size[1]

        self.Tilt_Prop = pygame.image.load(os.path.join(f"{img_path}/Tiltprop.png"))
        # self.Tilt_Prop = pygame.image.load(os.path.join("C:/Users/ds040/OneDrive - konkuk.ac.kr/Teams/RL/Transition-Train/images/Tiltprop.png"))
        Tilt_Prop_size = self.Tilt_Prop.get_rect().size
        self.Tilt_Prop_width = Tilt_Prop_size[0]
        self.Tilt_Prop_height = Tilt_Prop_size[1]
        
        self.Textboard = pygame.image.load(os.path.join(f"{img_path}/textboard(300x200).png"))
        # self.Textboard = pygame.image.load(os.path.join("C:/Users/ds040/OneDrive - konkuk.ac.kr/Teams/RL/Transition-Train/images/textboard.png"))
        
        pygame.font.init()
        self.font = pygame.font.SysFont('arial',20, True, True)  #폰트 설정
        
    def set_DB(self, DBname):

        DB = dataLoader('aero.json')
        
        self.cg_x       = DB["Configurations"]["cg_x"]                          # m
        self.cg_z       = DB["Configurations"]["cg_z"]                          # m
        self.f_p_x      = DB["Configurations"]["f_p_x"]                         # m
        self.f_p_z      = DB["Configurations"]["f_p_z"]                         # m
        self.r_p_x      = DB["Configurations"]["r_p_x"]                         # m
        self.r_p_z      = DB["Configurations"]["r_p_z"]                         # m
        self.aerorp_x   = DB["Configurations"]["aerorp_x"]                      # m
        self.aerorp_z   = DB["Configurations"]["aerorp_z"]                      # m
        self.S          = DB["Configurations"]["S"]                             # m^2
        self.cbar       = DB["Configurations"]["cbar"]                          # m
        self.elev_max   = DB["Configurations"]["elev_max"]                      # deg
        
        self.D_X_prop_f = self.cg_x - self.f_p_x                                # m
        self.D_Z_prop_f = self.cg_z - self.f_p_z                                # m
        self.D_X_prop_r = self.cg_x - self.r_p_x                                # m
        self.D_Z_prop_r = self.cg_z - self.r_p_z                                # m
        self.D_X_aero   = self.cg_x - self.aerorp_x                             # m
        self.D_Z_aero   = self.cg_z - self.aerorp_z                             # m
        
        self.K_T        = DB["Propulsion"]["K_T"]                               # none
        self.rpm_max    = DB["Propulsion"]["rpm_max"]                           # rpm
        self.tilt_min   = DB["Propulsion"]["tilt_min"]                          # rad
        self.tilt_max   = math.pi/2                                             # rad
        
        self.CL_a_20    = DB["Aerodynamics"]["CL"]["CL_a_20"]                   # none
        self.CL_a_15    = DB["Aerodynamics"]["CL"]["CL_a_15"]                   # none
        self.CL_a_10    = DB["Aerodynamics"]["CL"]["CL_a_10"]                   # none
        self.CL_a_5     = DB["Aerodynamics"]["CL"]["CL_a_5"]                    # none
        self.CL_a0      = DB["Aerodynamics"]["CL"]["CL_a0"]                     # none
        self.CL_a5      = DB["Aerodynamics"]["CL"]["CL_a5"]                     # none
        self.CL_a10     = DB["Aerodynamics"]["CL"]["CL_a10"]                    # none
        self.CL_a15     = DB["Aerodynamics"]["CL"]["CL_a15"]                    # none
        self.CL_a20     = DB["Aerodynamics"]["CL"]["CL_a20"]                    # none
        self.CL_a25     = DB["Aerodynamics"]["CL"]["CL_a25"]                    # none
        self.CL_a30     = DB["Aerodynamics"]["CL"]["CL_a30"]                    # none
        
        self.CD_a_20    = DB["Aerodynamics"]["CD"]["CD_a_20"]                   # none
        self.CD_a_15    = DB["Aerodynamics"]["CD"]["CD_a_15"]                   # none
        self.CD_a_10    = DB["Aerodynamics"]["CD"]["CD_a_10"]                   # none
        self.CD_a_5     = DB["Aerodynamics"]["CD"]["CD_a_5"]                    # none
        self.CD_a0      = DB["Aerodynamics"]["CD"]["CD_a0"]                     # none
        self.CD_a5      = DB["Aerodynamics"]["CD"]["CD_a5"]                     # none
        self.CD_a10     = DB["Aerodynamics"]["CD"]["CD_a10"]                    # none
        self.CD_a15     = DB["Aerodynamics"]["CD"]["CD_a15"]                    # none
        self.CD_a20     = DB["Aerodynamics"]["CD"]["CD_a20"]                    # none
        self.CD_a25     = DB["Aerodynamics"]["CD"]["CD_a25"]                    # none
        self.CD_a30     = DB["Aerodynamics"]["CD"]["CD_a30"]                    # none
        
        self.Cm_a_20    = DB["Aerodynamics"]["Cm"]["Cm_a_20"]                   # none
        self.Cm_a_15    = DB["Aerodynamics"]["Cm"]["Cm_a_15"]                   # none
        self.Cm_a_10    = DB["Aerodynamics"]["Cm"]["Cm_a_10"]                   # none
        self.Cm_a_5     = DB["Aerodynamics"]["Cm"]["Cm_a_5"]                    # none
        self.Cm_a0      = DB["Aerodynamics"]["Cm"]["Cm_a0"]                     # none
        self.Cm_a5      = DB["Aerodynamics"]["Cm"]["Cm_a5"]                     # none
        self.Cm_a10     = DB["Aerodynamics"]["Cm"]["Cm_a10"]                    # none
        self.Cm_a15     = DB["Aerodynamics"]["Cm"]["Cm_a15"]                    # none
        self.Cm_a20     = DB["Aerodynamics"]["Cm"]["Cm_a20"]                    # none
        self.Cm_a25     = DB["Aerodynamics"]["Cm"]["Cm_a25"]                    # none
        self.Cm_a30     = DB["Aerodynamics"]["Cm"]["Cm_a30"]                    # none
        
        self.CL_elev_0  = DB["Aerodynamics"]["CL_elev"]["CL_elev_0"]            # none
        self.CL_elev    = DB["Aerodynamics"]["CL_elev"]["CL_elev"]              # none/deg
        self.CD_elev_0  = DB["Aerodynamics"]["CL_elev"]["CD_elev_0"]            # none
        self.CD_elev    = DB["Aerodynamics"]["CL_elev"]["CD_elev"]              # none/deg
        self.Cm_elev_0  = DB["Aerodynamics"]["CL_elev"]["Cm_elev_0"]            # none
        self.Cm_elev    = DB["Aerodynamics"]["CL_elev"]["Cm_elev"]              # none/deg
        
        self.m          = DB["Aerodynamics"]["WnB"]["m"]                        # kg
        self.Iyy        = DB["Aerodynamics"]["WnB"]["Iyy"]                      # kg*m^2
        self.g          = DB["Aerodynamics"]["WnB"]["g"]                        # kg/m^2
        
    def set_init_state(self):
        self.state = [0, 0, 0, 0, 0, 0, 0]
                   # [X, Z, theta, U, W, q, time]
                   # [0, 1,     2, 3, 4, 5,    6]
        # self.action = [self.m*self.g/(4*self.K_T*self.rpm_max**2),       0.5,           1.0]
        #             # [                             Throttle_cmd, Pitch_cmd,      Tilt_cmd]
        self.f_rpm = round(math.sqrt(self.m*self.g/(4*self.K_T))/self.rpm_max, 3)
        self.r_rpm = round(math.sqrt(self.m*self.g/(4*self.K_T))/self.rpm_max, 3)
        self.elev = 0
        self.tilt = 90
        self.sim_time_dt = 0.05   
        self.rpm_rate = 0.01             # ratio (rpm/rpm_max)
        self.elev_rate = 1               # deg
        self.tilt_rate = 1               # deg
        # self.action = [self.f_rpm, self.r_rpm, self.elev, self.tilt]
                     # [ Front_rpm, Rear_rpm, Elevator, Tilt_angle]
        self.viewer = None
        self.current_score = 0
        
        self.T_f = 0
        self.T_r = 0
        self.L = 0
        self.D = 0
        self.Mp = 0
        self.al = 0
        # print(self.f_rpm)
        # print(self.r_rpm)
        
        SCREEN_COLOR = (255, 255, 255)
        if self.viewer == None:
            pygame.init()
            pygame.display.set_caption("Transition-Training")
            self.screen = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
        self.screen.fill(SCREEN_COLOR)
                
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    #################### __init__ ####################
          
    
    
    #################### reset ####################
    def reset(self):
        self.set_init_state()
        self.current_score = 0
        # observation = np.hstack((self.state, self.f_rpm, self.r_rpm, self.elev, self.tilt))  
        observation = np.hstack((self.state[0],self.state[1],self.state[2],self.state[3],self.state[4],self.state[5],self.state[6], self.f_rpm, self.r_rpm, self.elev, self.tilt))
        return observation
    #################### reset ####################
    
    
    
    
    
    #################### step ####################
    def step(self, action):
        
        # action_in = int(action)
        # action_in = int(27)
        # print(action_in)
        # print(self.current_score)
        # print(reward)
        (x1, x2, x3, x4) = action
        self.f_rpm   +=   x1*self.rpm_rate
        self.r_rpm   +=   x2*self.rpm_rate
        self.elev    +=   x3*self.elev_rate
        self.tilt    +=   x4*self.tilt_rate

        self.f_rpm = np.max([0.0, self.f_rpm])
        self.f_rpm = np.min([1.0, self.f_rpm])

        self.r_rpm = np.max([0.0, self.r_rpm])
        self.r_rpm = np.min([1.0, self.r_rpm])      
        
        self.Simulation()
        
        # Reward 계산 | Reward Calculation | 보상 계산
        pitch_state = abs(self.state[2])*180/math.pi # degree
        # print(f"pitch state: {pitch_state}")
        # print(f"pitch state: {pitch_state}")
        alt_state = self.state[1]

        # 조건 1: tilt각 차이 | self.tilt 초기값: 90 deg | 작을수록 좋음 | 0~90
        # 90으로부터 멀어지면 좋음
        # reward 1은 클수록 좋게 설정하였음(positive)
        reward_1 = (90 - self.tilt) / 90

        # 조건 2: 피치 값 차이 | 작을 수록 좋음 | 0~90
        # reward 2는 클수록 좋게 설정하였음(positive)
        pitch_target = 0.0
        if 0 <= pitch_state<= 8:
            reward_2 = (10 - np.abs(pitch_state - pitch_target)) / 10
        else:
            reward_2 = -1
        
        # 조건 3: 비행 시간 | 클수록 좋음 | 0 ~ inf, 1 timestep = 0.05 sec, 30,000 timestep = 1,500 sec = 25 min
        # 조건 3은 클수록 좋게 설정하였음(positive)
        reward_3 = self.state[6]
        
        # 조건 4: 크루즈 속도 차이 | 작을 수록 좋음 | 0~20
        # 조건 4는 클수록 좋게 설정하였음(positive)
        Vcruise_target = 30
        Vcruise_limit = 50
        if Vcruise_target <=Vcruise_limit:
            reward_4 = (Vcruise_target - np.abs(self.state[3] - Vcruise_target)) / Vcruise_target
        else:
            reward_4 = -1

        # 조건 5: 순항 고도 | 초기 고도: 0m > 15m나 0m나 대기 조건 차이 크지 않음 | 작을 수록 좋음 | -15 ~ +15
        # 조건 5는 클수록 좋게 설정하였음(positive)
        reward_5 = (15 - np.abs(self.state[1])) / 15

        # 조건 6: 프로펠러 rpm 최소화 | 작을수록 좋음 | 0 ~ 1
        # 조건 6: 프로펠러 rpm 최소화 | 작을수록 좋음 | 0 ~ 1
        # 조건 6은 클수록 좋게 설정하였음(Positive)
        reward_6 = (1 - self.r_rpm) + (1 - self.f_rpm)

        # 조건 7: 이동 거리
        reward_7 = self.state[0]

        # 비행 속도가 20 미만일 경우와 20 이상일 경우 가중치를 다르게 배정
        # [틸트각, 피치, 비행 시간, 비행 속도, 순항 고도, 프로펠러 rpm, 이동 거리]
        # 8/19: 분기점을 10m/s로 조정
        if self.state[3] < 10:
            weight = [500, 100, 20, 100, 100, 100, 2]
        else:
            weight = [500, 100, 1, 200, 200, 50, 1]

        rewards_list = [reward_1, reward_2, reward_3, reward_4, reward_5, reward_6, reward_7]
        reward = np.dot(weight, rewards_list)
        
        # Sharp reward(editing)
        done = False
        
        alt_constrain = 15
        pitch_constrain = 15
        
        if (np.abs(alt_state)  >= alt_constrain):
            done = True
        if (np.abs(pitch_state)  >= pitch_constrain):
            done = True
        
        if (self.tilt < 0 or self.tilt > 90):
            done = True

        if self.state[3] > Vcruise_limit:
            done = True
             
        observation = np.hstack((self.state[0],self.state[1],self.state[2],self.state[3],self.state[4],self.state[5],self.state[6], self.f_rpm, self.r_rpm, self.elev, self.tilt))
        reward_detail = [reward_1, reward_2, reward_3, reward_4, reward_5, reward_6, reward_7]
        step_data = self.dataCollection()
        info = {
            'Time': self.state[6],
            'x_pos': self.state[0],
            'z_pos': self.state[1],
            'pitch': self.state[2],
           # 'Tilt': self.action[2]
            'reward_detail': reward_detail,
            'data': step_data
        }
        return observation, reward, done, info
    
    # Flight Dynamics Equations
    # def fqdot(self, q):
    #     return ((-self.f_Lz*math.cos(self.tilt * math.pi/180) + self.f_Lx*math.sin(self.tilt * math.pi/180))*self.T_f - self.r_Lx*self.T_r + self.Mp)/self.Iyy
    
    # def fudot(self, u):
    #     return -self.g*math.sin(self.state[2]) - self.state[5]*self.state[4] + (self.T_f*math.cos(self.tilt * math.pi/180) - self.D*math.cos(self.al) - self.L*math.sin(self.al))/self.m
    
    # def fwdot(self, w):
    #     return self.g*math.cos(self.state[2]) + self.state[5]*self.state[3] + (- self.T_f*math.sin(self.tilt * math.pi/180) - self.T_r + self.D*math.sin(self.al) - self.L*math.cos(self.al))/self.m
    
    def fqdot(self, q):
        return self.Myb/self.Iyy
    
    def fudot(self, u):
        return self.Fxb/self.m
    
    def fwdot(self, w):
        return self.Fzb/self.m
    
    def fthedot(self, the):
        return self.q
    
    def fxdot(self, x):
        return self.u*math.cos(self.state[2]) + self.w*math.sin(self.state[2])
    
    def fzdot(self, z):
        return -self.u*math.sin(self.state[2]) + self.w*math.cos(self.state[2])
    
    # Simulation
    def Simulation(self):
        self.x = self.state[0]
        self.z = self.state[1]
        self.the = self.state[2]
        self.u = self.state[3]
        self.w = self.state[4]
        self.q = self.state[5]
        t = self.state[6]
        
        # =============== Vehicle Model (Calculate Force&Moments) ===============
        self.T_f = 2*self.K_T*(self.f_rpm*self.rpm_max)**2
        self.T_r = 2*self.K_T*(self.r_rpm*self.rpm_max)**2
        
        
        if self.state[3] == 0:
            self.al = self.state[2]
        else:
            self.al = math.atan(self.w/self.u)
        
        
        
        self.vel = math.sqrt(self.w**2 + self.u**2)
        
        # ckchoi: 추정식 잘못되어서 수정
        # guess_value = ((d-b)/(c-a))*(x-a)+b
        if (self.al <= -15*math.pi/180):
            CL_clean = self.CL_a_20 + (self.al*180/math.pi + 20)*(self.CL_a_15 - self.CL_a_20)/5
            CD_clean = self.CD_a_20 + (self.al*180/math.pi + 20)*(self.CD_a_15 - self.CD_a_20)/5
            Cm_clean = self.Cm_a_20 + (self.al*180/math.pi + 20)*(self.Cm_a_15 - self.Cm_a_20)/5
            CL_clean = self.CL_a_20 + (self.al*180/math.pi + 20)*(self.CL_a_15 - self.CL_a_20)/5
            CD_clean = self.CD_a_20 + (self.al*180/math.pi + 20)*(self.CD_a_15 - self.CD_a_20)/5
            Cm_clean = self.Cm_a_20 + (self.al*180/math.pi + 20)*(self.Cm_a_15 - self.Cm_a_20)/5
        elif (self.al <= -10*math.pi/180):
            CL_clean = self.CL_a_15 + (self.al*180/math.pi + 15)*(self.CL_a_10 - self.CL_a_15)/5
            CD_clean = self.CD_a_15 + (self.al*180/math.pi + 15)*(self.CD_a_10 - self.CD_a_15)/5
            Cm_clean = self.Cm_a_15 + (self.al*180/math.pi + 15)*(self.Cm_a_10 - self.Cm_a_15)/5
            CL_clean = self.CL_a_15 + (self.al*180/math.pi + 15)*(self.CL_a_10 - self.CL_a_15)/5
            CD_clean = self.CD_a_15 + (self.al*180/math.pi + 15)*(self.CD_a_10 - self.CD_a_15)/5
            Cm_clean = self.Cm_a_15 + (self.al*180/math.pi + 15)*(self.Cm_a_10 - self.Cm_a_15)/5
        elif (self.al <= -5*math.pi/180):
            CL_clean = self.CL_a_10 + (self.al*180/math.pi + 10)*(self.CL_a_5 - self.CL_a_10)/5
            CD_clean = self.CD_a_10 + (self.al*180/math.pi + 10)*(self.CD_a_5 - self.CD_a_10)/5
            Cm_clean = self.Cm_a_10 + (self.al*180/math.pi + 10)*(self.Cm_a_5 - self.Cm_a_10)/5
            CL_clean = self.CL_a_10 + (self.al*180/math.pi + 10)*(self.CL_a_5 - self.CL_a_10)/5
            CD_clean = self.CD_a_10 + (self.al*180/math.pi + 10)*(self.CD_a_5 - self.CD_a_10)/5
            Cm_clean = self.Cm_a_10 + (self.al*180/math.pi + 10)*(self.Cm_a_5 - self.Cm_a_10)/5
        elif (self.al <= 0*math.pi/180):
            CL_clean = self.CL_a_5 + (self.al*180/math.pi + 5)*(self.CL_a0 - self.CL_a_5)/5
            CD_clean = self.CD_a_5 + (self.al*180/math.pi + 5)*(self.CD_a0 - self.CD_a_5)/5
            Cm_clean = self.Cm_a_5 + (self.al*180/math.pi + 5)*(self.Cm_a0 - self.Cm_a_5)/5
            CL_clean = self.CL_a_5 + (self.al*180/math.pi + 5)*(self.CL_a0 - self.CL_a_5)/5
            CD_clean = self.CD_a_5 + (self.al*180/math.pi + 5)*(self.CD_a0 - self.CD_a_5)/5
            Cm_clean = self.Cm_a_5 + (self.al*180/math.pi + 5)*(self.Cm_a0 - self.Cm_a_5)/5
        elif (self.al <= 5*math.pi/180):
            CL_clean = self.CL_a0 + (self.al*180/math.pi - 0)*(self.CL_a5 - self.CL_a0)/5
            CD_clean = self.CD_a0 + (self.al*180/math.pi - 0)*(self.CD_a5 - self.CD_a0)/5
            Cm_clean = self.Cm_a0 + (self.al*180/math.pi - 0)*(self.Cm_a5 - self.Cm_a0)/5
            CL_clean = self.CL_a0 + (self.al*180/math.pi - 0)*(self.CL_a5 - self.CL_a0)/5
            CD_clean = self.CD_a0 + (self.al*180/math.pi - 0)*(self.CD_a5 - self.CD_a0)/5
            Cm_clean = self.Cm_a0 + (self.al*180/math.pi - 0)*(self.Cm_a5 - self.Cm_a0)/5
        elif (self.al <= 10*math.pi/180):
            CL_clean = self.CL_a5 + (self.al*180/math.pi - 5)*(self.CL_a10 - self.CL_a5)/5
            CD_clean = self.CD_a5 + (self.al*180/math.pi - 5)*(self.CD_a10 - self.CD_a5)/5
            Cm_clean = self.Cm_a5 + (self.al*180/math.pi - 5)*(self.Cm_a10 - self.Cm_a5)/5
            CL_clean = self.CL_a5 + (self.al*180/math.pi - 5)*(self.CL_a10 - self.CL_a5)/5
            CD_clean = self.CD_a5 + (self.al*180/math.pi - 5)*(self.CD_a10 - self.CD_a5)/5
            Cm_clean = self.Cm_a5 + (self.al*180/math.pi - 5)*(self.Cm_a10 - self.Cm_a5)/5
        elif (self.al <= 15*math.pi/180):
            CL_clean = self.CL_a10 + (self.al*180/math.pi - 10)*(self.CL_a15 - self.CL_a10)/5
            CD_clean = self.CD_a10 + (self.al*180/math.pi - 10)*(self.CD_a15 - self.CD_a10)/5
            Cm_clean = self.Cm_a10 + (self.al*180/math.pi - 10)*(self.Cm_a15 - self.Cm_a10)/5
            CL_clean = self.CL_a10 + (self.al*180/math.pi - 10)*(self.CL_a15 - self.CL_a10)/5
            CD_clean = self.CD_a10 + (self.al*180/math.pi - 10)*(self.CD_a15 - self.CD_a10)/5
            Cm_clean = self.Cm_a10 + (self.al*180/math.pi - 10)*(self.Cm_a15 - self.Cm_a10)/5
        elif (self.al <= 20*math.pi/180):
            CL_clean = self.CL_a15 + (self.al*180/math.pi - 15)*(self.CL_a20 - self.CL_a15)/5
            CD_clean = self.CD_a15 + (self.al*180/math.pi - 15)*(self.CD_a20 - self.CD_a15)/5
            Cm_clean = self.Cm_a15 + (self.al*180/math.pi - 15)*(self.Cm_a20 - self.Cm_a15)/5
            CL_clean = self.CL_a15 + (self.al*180/math.pi - 15)*(self.CL_a20 - self.CL_a15)/5
            CD_clean = self.CD_a15 + (self.al*180/math.pi - 15)*(self.CD_a20 - self.CD_a15)/5
            Cm_clean = self.Cm_a15 + (self.al*180/math.pi - 15)*(self.Cm_a20 - self.Cm_a15)/5
        elif (self.al <= 25*math.pi/180):
            CL_clean = self.CL_a20 + (self.al*180/math.pi - 20)*(self.CL_a25 - self.CL_a20)/5
            CD_clean = self.CD_a20 + (self.al*180/math.pi - 20)*(self.CD_a25 - self.CD_a20)/5
            Cm_clean = self.Cm_a20 + (self.al*180/math.pi - 20)*(self.Cm_a25 - self.Cm_a20)/5
            CL_clean = self.CL_a20 + (self.al*180/math.pi - 20)*(self.CL_a25 - self.CL_a20)/5
            CD_clean = self.CD_a20 + (self.al*180/math.pi - 20)*(self.CD_a25 - self.CD_a20)/5
            Cm_clean = self.Cm_a20 + (self.al*180/math.pi - 20)*(self.Cm_a25 - self.Cm_a20)/5
        else:
            CL_clean = self.CL_a25 + (self.al*180/math.pi - 25)*(self.CL_a30 - self.CL_a25)/5
            CD_clean = self.CD_a25 + (self.al*180/math.pi - 25)*(self.CD_a30 - self.CD_a25)/5
            Cm_clean = self.Cm_a25 + (self.al*180/math.pi - 25)*(self.Cm_a30 - self.Cm_a25)/5
        
        CL_CS = self.CL_elev_0 + self.CL_elev*self.elev
        CD_CS = self.CD_elev_0 + self.CD_elev*self.elev
        Cm_CS = self.Cm_elev_0 + self.Cm_elev*self.elev
        
        if (self.al >= (-20*math.pi/180)) and (self.al <= (30*math.pi/180)):
            self.CL = (CL_clean + CL_CS) # 항공기 전체 양력 계수입니다.
            self.CD = (CD_clean + CD_CS) # 항공기 전체 항력 계수입니다.

            self.L = 0.5 * 1.225 * (self.vel**2) * self.S * (CL_clean + CL_CS)
            self.D = 0.5 * 1.225 * (self.vel**2) * self.S * (CD_clean + CD_CS)
            self.Mp = 0.5 * 1.225 * (self.vel**2) * self.S * self.cbar * (Cm_clean + Cm_CS)
            # print(f"Lift: {self.L} | Velocity: {self.vel} | Wing Area: {self.S} | Cl:{(CL_clean)} | pitch: {self.al} vs {self.state[2]}")
        else:
            self.L = 0
            self.D = 0
            self.Mp = 0
        
        self.Myb = self.D_Z_prop_f*self.T_f*math.cos(math.radians(self.tilt)) + self.D_X_prop_f*self.T_f*math.sin(math.radians(self.tilt)) + self.D_X_prop_r*self.T_r - self.D_Z_aero*(self.D*math.cos(math.radians(self.al)) + self.L*math.sin(math.radians(self.al))) - self.D_X_aero*(self.D*math.sin(math.radians(self.al)) - self.L*math.cos(math.radians(self.al))) + self.Mp
        self.Fxb = self.T_f*math.cos(math.radians(self.tilt)) - self.D*math.cos(math.radians(self.al)) - self.L*math.sin(math.radians(self.al)) + self.m*self.g*math.sin(math.radians(self.al))
        self.Fzb = -self.T_r - self.T_f*math.sin(math.radians(self.tilt)) + self.D*math.sin(math.radians(self.al)) - self.L*math.cos(math.radians(self.al)) + self.m*self.g*math.cos(math.radians(self.al))
        # =============== Vehicle Model (Calculate Force&Moments) ===============
        
        # =============== Flight Dynamics with RK-4 (Calculate Next Status) ===============

        # q
        k1 = self.sim_time_dt * (self.fqdot(self.q))
        k2 = self.sim_time_dt * (self.fqdot(self.q + k1/2))
        k3 = self.sim_time_dt * (self.fqdot(self.q + k2/2))
        k4 = self.sim_time_dt * (self.fqdot(self.q + k3))
        k = (k1 + 2*k2 + 2*k3 + k4)/6
        qn = self.q + k
        
        self.q = qn
        
        # u & ax
        k1 = self.sim_time_dt * (self.fudot(self.u))
        k2 = self.sim_time_dt * (self.fudot((self.u + k1/2)))
        k3 = self.sim_time_dt * (self.fudot((self.u + k2/2)))
        k4 = self.sim_time_dt * (self.fudot((self.u + k3)))
        k = (k1 + 2*k2 + 2*k3 + k4)/6
        un = self.u + k
        self.ax = k/self.sim_time_dt
        
        # w & az
        k1 = self.sim_time_dt * (self.fwdot(self.w))
        k2 = self.sim_time_dt * (self.fwdot((self.w + k1/2)))
        k3 = self.sim_time_dt * (self.fwdot((self.w + k2/2)))
        k4 = self.sim_time_dt * (self.fwdot((self.w + k3)))
        k = (k1+2*k2+2*k3+k4)/6
        wn = self.w + k
        self.az = k/self.sim_time_dt
        
        self.u = un
        self.w = wn
        
        # the
        k1 = self.sim_time_dt * (self.fthedot(self.the))
        k2 = self.sim_time_dt * (self.fthedot((self.the + k1/2)))
        k3 = self.sim_time_dt * (self.fthedot((self.the + k2/2)))
        k4 = self.sim_time_dt * (self.fthedot((self.the + k3)))
        k = (k1 + 2*k2 + 2*k3 + k4)/6
        then = self.the + k
        
        self.the = then
        
        # x
        k1 = self.sim_time_dt * (self.fxdot(self.x))
        k2 = self.sim_time_dt * (self.fxdot((self.x + k1/2)))
        k3 = self.sim_time_dt * (self.fxdot((self.x + k2/2)))
        k4 = self.sim_time_dt * (self.fxdot((self.x + k3)))
        k = (k1 + 2*k2 + 2*k3 + k4)/6
        xn = self.x + k
        
        # z
        k1 = self.sim_time_dt * (self.fzdot(self.z))
        k2 = self.sim_time_dt * (self.fzdot((self.z + k1/2)))
        k3 = self.sim_time_dt * (self.fzdot((self.z + k2/2)))
        k4 = self.sim_time_dt * (self.fzdot((self.z + k3)))
        k = (k1 + 2*k2 + 2*k3 + k4)/6
        zn = self.z + k
        
        self.x = xn
        self.z = zn

        self.Sim_time = t + self.sim_time_dt

        self.state = [self.x, self.z, self.the, self.u, self.w, self.q, self.Sim_time]
        # =============== Flight Dynamics with RK-4 (Calculate Next Status) ===============
    #################### step ####################
    
    #################### data collection ####################
    def dataCollection(self):
        data = {
            "time": self.Sim_time,
            "CL": self.CL,
            "CD": self.CD,
            "X": self.state[0],
            "Z": self.state[1],
            "theta": self.the,
            "f_rpm": self.f_rpm,
            "r_rpm": self.r_rpm,
            "U": self.u,
            "tilt": self.tilt,
            "Lift": self.L,
            "L/W": round(self.L/(self.m*self.g), 8),
            "(L+T)/W": round((self.L + self.T_f + self.T_r)/(self.m*self.g), 8)
        }
        return data
    #################### data collection ####################

    #################### render ####################
    def render(self, mode='human'):
        self.draw_vehicle(self.state[0], self.state[1], self.state[2], self.tilt*math.pi/180)
        
        text_x = self.font.render("Distance(m): "+str(round(self.state[0],8)),True,(28,0,0))
        text_z = self.font.render("Altitude(m): "+str(round(-self.state[1],8)),True,(28,0,0))
        text_pitch = self.font.render("Pitch(deg): "+str(round(self.state[2]*180.0/math.pi,8)),True,(28,0,0))
        text_u = self.font.render("U(m/s): "+str(round(self.state[3],8)),True,(28,0,0))
        text_w = self.font.render("W(m/s): "+str(round(-self.state[4],8)),True,(28,0,0))
        text_time = self.font.render("Time(sec) : "+str(round(self.state[6],8)),True,(28,0,0))
        text_f_rpm = self.font.render("Front_RPM(%) : "+str(round(self.f_rpm*100,8)),True,(28,0,0))
        text_r_rpm = self.font.render("Rear_RPM(%) : "+str(round(self.r_rpm*100,8)),True,(28,0,0))
        text_elev = self.font.render("Elevator(deg) : "+str(round(self.elev,8)),True,(28,0,0))
        text_tilt = self.font.render("Tilt_Angle(deg) : "+str(round(self.tilt,8)),True,(28,0,0))
        text_Tf = self.font.render("Front Thrust(N) : "+str(round(self.T_f,8)),True,(28,0,0))
        text_Tr = self.font.render("Rear Thrust(N) : "+str(round(self.T_r,8)),True,(28,0,0))
        text_W = self.font.render("Weight(N) : "+str(round(self.m*self.g,8)),True,(28,0,0))
        # ckchoi(8.24): 원본 코드에서 양력에 100을 곱했는데 제거함
        # original: text_L = self.font.render("Lift(N) : "+str(round(self.L*100,8)),True,(28,0,0))
        text_L = self.font.render("Lift(N) : "+str(round(self.L,8)),True,(28,0,0))
        text_D = self.font.render("Drag(N) : "+str(round(self.D,8)),True,(28,0,0))
        text_LperD = self.font.render(f"Lift/Weight: {str(round(self.L/(self.m*self.g),8))}", True, (28,0,0))
        text_LTperD = self.font.render(f"Lift+Thrust/Weight: {str(round((self.L + self.T_f + self.T_r)/(self.m*self.g),8))}", True, (28,0,0))
        text_M = self.font.render("Pithching(N) : "+str(round(self.Mp,8)),True,(28,0,0))
        text_alpha = self.font.render("AoA(deg) : "+str(round(self.al,1)),True,(28,0,0))
        
        self.screen.blit(self.Textboard, (700, 10))
        self.screen.blit(text_x, (700,10))
        self.screen.blit(text_z, (700,30))
        self.screen.blit(text_pitch, (700,50))
        self.screen.blit(text_u, (700,70))
        self.screen.blit(text_w, (700,90))
        self.screen.blit(text_time, (700,110))
        self.screen.blit(text_f_rpm, (700,130))
        self.screen.blit(text_r_rpm, (700,150))
        self.screen.blit(text_elev, (700,170))
        self.screen.blit(text_tilt, (700,190))
        self.screen.blit(text_Tf, (700,210))
        self.screen.blit(text_Tr, (700,230))
        self.screen.blit(text_W, (700,250))
        self.screen.blit(text_L, (700,270))
        self.screen.blit(text_LperD, (700,290))
        self.screen.blit(text_LTperD, (700,310))
        self.screen.blit(text_D, (700,330))
        self.screen.blit(text_M, (700,350))
        self.screen.blit(text_alpha, (700,370))
        
        self.clock.tick(120)
        pygame.display.flip() 
    
    
    def draw_vehicle(self, x, z, the, tilt):
        rotated_image_vehicle = pygame.transform.rotate(self.vehicle,((the)*180/math.pi))
        rotated_image_Tilt_Prop = pygame.transform.rotate(self.Tilt_Prop, ((the + tilt - math.pi/2)*180/math.pi))   
    
        f_tilt_rotate_pos_x = (270 - self.vehicle_width/2)
        f_tilt_rotate_pos_z = (self.vehicle_height/2 - 153)
        f_tilt_rotate_pos_length = math.sqrt(f_tilt_rotate_pos_x**2 + f_tilt_rotate_pos_z**2)
        
        r_tilt_rotate_pos_x = (121 - self.vehicle_width/2)
        r_tilt_rotate_pos_z = (self.vehicle_height/2 - 153)
        r_tilt_rotate_pos_length = math.sqrt(r_tilt_rotate_pos_x**2 + r_tilt_rotate_pos_z**2)
        
        vehicle_f_tilt_angle = math.atan(f_tilt_rotate_pos_z/f_tilt_rotate_pos_x)
        
        vehicle_r_tilt_angle = math.atan(r_tilt_rotate_pos_z/r_tilt_rotate_pos_x)
        
        #Drawing vectors of motor forces
        
        if x <= 200:
            self.screen.blit(self.background, (-x, 0)) 
        else:
            self.screen.blit(self.background, (-x - 200, 0)) #-x
            
        if x <= 200:
            self.screen.blit(rotated_image_vehicle, ((200 + x - rotated_image_vehicle.get_width()/2), (self.window_size[1]/2 + z - rotated_image_vehicle.get_height()/2)))
            
            self.screen.blit(rotated_image_Tilt_Prop, (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + the) \
                                                       
                                             + (self.Tilt_Prop_height/2)*math.cos(vehicle_f_tilt_angle + the + tilt) - rotated_image_Tilt_Prop.get_width()/2,\
                                                 
                                             self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + the) - (self.Tilt_Prop_height/2)*math.sin(vehicle_f_tilt_angle + the + tilt)\
                                                 
                                             - rotated_image_Tilt_Prop.get_height()/2))
            
            pygame.draw.line(self.screen, (179,179,179),\
                             
                             (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + the), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + the)),\
                             
                             (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + the) + 50*math.cos(the + tilt), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + the) - 50*math.sin(the + tilt)), 4)
            
            pygame.draw.line(self.screen, (255,0,0),\
                             
                             (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + the), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + the)),\
                             
                             (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + the) + self.f_rpm*50*math.cos(the + tilt), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + the) - self.f_rpm*50*math.sin(the + tilt)), 4)
            
            pygame.draw.line(self.screen, (179,179,179),\
                             
                             (200 + x - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + the), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + the)),\
                             
                             (200 + x - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + the) + 50*math.cos(the + math.pi/2), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + the) - 50*math.sin(the + math.pi/2)), 4)
            
            pygame.draw.line(self.screen, (255,0,0),\
                             
                             (200 + x - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + the), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + the)),\
                             
                             (200 + x - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + the) + self.r_rpm*50*math.cos(the + math.pi/2), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + the) - self.r_rpm*50*math.sin(the + math.pi/2)), 4)
                
            
        else:   
            self.screen.blit(rotated_image_vehicle, ((400 - rotated_image_vehicle.get_width()/2), (self.window_size[1]/2 + z - rotated_image_vehicle.get_height()/2)))
            
            self.screen.blit(rotated_image_Tilt_Prop, (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + the)\
                                                       
                                             + (self.Tilt_Prop_height/2)*math.cos(vehicle_f_tilt_angle + the + tilt) - rotated_image_Tilt_Prop.get_width()/2,\
                                                 
                                             self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + the) - (self.Tilt_Prop_height/2)*math.sin(vehicle_f_tilt_angle + the + tilt)\
                                                 
                                             - rotated_image_Tilt_Prop.get_height()/2))
            
            pygame.draw.line(self.screen, (179,179,179),\
                             
                             (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + the), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + the)),\
                             
                             (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + the) + 50*math.cos(the + tilt), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + the) - 50*math.sin(the + tilt)), 4)
            
            pygame.draw.line(self.screen, (255,0,0),\
                             
                             (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + the), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + the)),\
                             
                             (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + the) + self.f_rpm*50*math.cos(the + tilt), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + the) - self.f_rpm*50*math.sin(the + tilt)), 4)
            
            pygame.draw.line(self.screen, (179,179,179),\
                             
                             (400 - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + the), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + the)),\
                             
                             (400 - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + the) + 50*math.cos(the + math.pi/2), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + the) - 50*math.sin(the + math.pi/2)), 4)
            
            pygame.draw.line(self.screen, (255,0,0),\
                             
                             (400 - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + the), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + the)),\
                             
                             (400 - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + the) + self.r_rpm*50*math.cos(the + math.pi/2), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + the) - self.r_rpm*50*math.sin(the + math.pi/2)), 4)
                
    #################### render ####################

    #################### close ####################
    def close(self):
        if self.viewer != None:
            pygame.quit()
    ################### close ####################