import gym
import math
import pygame
import os
import numpy as np
from gym import spaces
from gym.utils import seeding

from src.loadDB import dataLoader
from src.Functions import linearInterpolation

class TiltrotorTransitionSimulator(gym.Env):
    metadata = {'render.modes': ['human']}
    ################### __init__ ####################
    def __init__(self):
        self.set_render(window_size=[1000,500])
        
        self.setSimulationData("aero.json")
        
        self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(12,), dtype=np.float32)
        
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
        
    def setSimulationData(self, AeroDBname='aero.json', constrainDBname='constrains.json'):

        SIMDB = dataLoader(AeroDBname)
        CONSTDB = dataLoader(constrainDBname)
        
        self.cg_x            = SIMDB["Configurations"]["cg_x"]                 # m
        self.cg_z            = SIMDB["Configurations"]["cg_z"]                 # m
        self.frontProp_x     = SIMDB["Configurations"]["frontProp_x"]          # m
        self.frontProp_z     = SIMDB["Configurations"]["frontProp_z"]          # m
        self.rearProp_x      = SIMDB["Configurations"]["rearProp_x"]           # m
        self.rearProp_z      = SIMDB["Configurations"]["rearProp_z"]           # m
        self.aeroCenter_x    = SIMDB["Configurations"]["aeroCenter_x"]         # m
        self.aeroCenter_z    = SIMDB["Configurations"]["aeroCenter_z"]         # m
        self.S               = SIMDB["Configurations"]["S"]                    # m^2
        self.cbar            = SIMDB["Configurations"]["cbar"]                 # m
        self.elevMaxDeg      = SIMDB["Configurations"]["elev_max"]             # deg
        self.stallSpeed      = SIMDB["Configurations"]["stall_speed"]          # m/s

        self.cg2FrontProp_x  = self.cg_x - self.frontProp_x                    # m
        self.cg2FrontProp_z  = self.cg_z - self.frontProp_z                    # m
        self.cg2RearProp_x   = self.cg_x - self.rearProp_x                     # m
        self.cg2RearProp_z   = self.cg_z - self.rearProp_z                     # m
        self.cg2AeroCenter_x = self.cg_x - self.aeroCenter_x                   # m
        self.cg2AeroCenter_z = self.cg_z - self.aeroCenter_z                   # m

        self.K_T             = SIMDB["Propulsion"]["K_T"]                      # none
        self.rpm_max         = SIMDB["Propulsion"]["rpm_max"]                  # rpm
        self.tilt_min        = SIMDB["Propulsion"]["tilt_min"]                 # rad
        self.tilt_max        = SIMDB["Propulsion"]["tilt_max"]                 # rad

        self.CL_table = {}
        self.CD_table = {}
        self.Cm_table = {}
        for aoa in range(-20, 31, 5):
            self.CL_table[str(aoa)] = SIMDB["Aerodynamics"]["CL"][str(aoa)]
            self.CD_table[str(aoa)] = SIMDB["Aerodynamics"]["CD"][str(aoa)]
            self.Cm_table[str(aoa)] = SIMDB["Aerodynamics"]["Cm"][str(aoa)]
        
        self.elev_CL_0       = SIMDB["Aerodynamics"]["elev"]["elev_CL_0"]      # none
        self.elev_CL_slop    = SIMDB["Aerodynamics"]["elev"]["elev_CL_slop"]   # none/deg
        self.elev_CD_0       = SIMDB["Aerodynamics"]["elev"]["elev_CD_0"]      # none
        self.elev_CD_slop    = SIMDB["Aerodynamics"]["elev"]["elev_CD_slop"]   # none/deg
        self.elev_Cm_0       = SIMDB["Aerodynamics"]["elev"]["elev_Cm_0"]      # none
        self.elev_Cm_slop    = SIMDB["Aerodynamics"]["elev"]["elev_Cm_slop"]   # none/deg
        
        self.m               = SIMDB["WeightAndBalance"]["m"]                  # kg
        self.Iyy             = SIMDB["WeightAndBalance"]["Iyy"]                # kg*m^2
        self.g               = SIMDB["WeightAndBalance"]["g"]                  # kg/m^2

        ## Constrains Data
        self.altitudeDelta   = CONSTDB["Constrains"]["altitudeDelta"]          # m
        self.VcruiseMax      = CONSTDB["Constrains"]["VcruiseMax"]             # m/s
        self.pitchMin        = CONSTDB["Constrains"]["pitchMin"]               # deg
        self.pitchMax        = CONSTDB["Constrains"]["pitchMax"]               # deg
        self.gForceMax       = CONSTDB["Constrains"]["g-forceMax"]             # g

        ## Target Data
        self.VcruiseTarget   = CONSTDB["Target"]["Vcruise"]                    # m/s
        self.pitchTarget     = CONSTDB["Target"]["pitch"]                      # deg
        self.gForceTarget    = CONSTDB["Target"]["g-force"]                    # g
        
    def set_init_state(self):
        self.state = [0] * 8
                   # [0: x, 1: z, 2: theta, 3: U, 4: W, 5: q, 6: time, 7: gForce]
        
        self.frontRPM = round(math.sqrt(self.m*self.g/(4*self.K_T))/self.rpm_max, 3)
        self.rearRPM = round(math.sqrt(self.m*self.g/(4*self.K_T))/self.rpm_max, 3)
        self.elev_deg = 0
        self.tilt_deg = 90

        self.time_delta = 0.05   
        self.rpm_rate = 0.01             # ratio (rpm/rpm_max)
        self.elev_rate = 1               # deg
        self.tilt_rate = 1               # deg
        
        self.viewer = None
        self.current_score = 0
        
        self.frontThrust = 0
        self.rearThrust = 0
        self.L = 0
        self.D = 0
        self.Mp = 0
        self.aoa = 0                     # radians
        self.gForce = 0
                
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
        
        observation = np.hstack((self.state[0],self.state[1],self.state[2],self.state[3],
                                 self.state[4],self.state[5],self.state[6],self.state[7],
                                 self.frontRPM,   self.rearRPM,   self.elev_deg,    self.tilt_deg))
        return observation
    #################### reset ####################    
    
    def calculateReward(self):
        # 조건 1: tilt각 차이 | tilt_deg가 0일 경우 최대(1) | tilt_deg가 작을 수록 좋음
        reward_tilt = (self.tilt_max - self.tilt_deg) / self.tilt_max

        # 조건 2: 피치 값 차이 | 피치가 pitchTarget일 경우 최대(1) | pitch가 pitchTarget에 가까울 수록 좋음
        pitch_deg = math.degrees(self.state[2])
        pitch_bandwidth = self.pitchMax - self.pitchMin
        if self.pitchMin <= pitch_deg <= self.pitchTarget:
            slop = 2 * pitch_bandwidth / (self.pitchTarget - self.pitchMin)
            reward_pitch = (pitch_bandwidth - slop * abs(pitch_deg - self.pitchTarget)) / pitch_bandwidth
        elif self.pitchTarget < pitch_deg <= self.pitchMax:
            slop = 2 * pitch_bandwidth / (self.pitchMax - self.pitchTarget)
            reward_pitch = (pitch_bandwidth - slop * abs(pitch_deg - self.pitchTarget)) / pitch_bandwidth
        else:
            reward_pitch = -1

        # 조건 3: 비행 시간 | 클수록 좋음 | 0 ~ inf, 1 timestep = 0.05 sec, 30,000 timestep = 1,500 sec = 25 min
        reward_time = self.state[6]

        # 조건 4: 크루즈 속도 차이 | speed가 VcruiseTarget일 경우 최대(1) | speed가 VcruiseTarget에 가까울 수록 좋음
        speed_bandwidth = self.VcruiseMax - self.VcruiseTarget
        if 0 <= self.state[3] <= self.VcruiseTarget:
            slop = 2 * speed_bandwidth / self.VcruiseTarget
            reward_speed = (speed_bandwidth - slop * abs(self.state[3] - self.VcruiseTarget)) / speed_bandwidth
        elif self.VcruiseTarget < self.state[3] <= self.VcruiseMax:
            slop = 2 * speed_bandwidth / (self.VcruiseMax - self.VcruiseTarget)
            reward_speed = (speed_bandwidth - slop * abs(self.state[3] - self.VcruiseTarget)) / speed_bandwidth
        else:
            reward_speed = -1
        
        # 조건 5: 순항 고도 | 초기 고도: 0m > 15m나 0m나 대기 조건 차이 크지 않음 | 작을 수록 좋음 | -15 ~ +15
        reward_altitude = (self.altitudeDelta - abs(self.state[1])) / self.altitudeDelta

        # 조건 6: 프로펠러 rpm 최소화 | 작을수록 좋음 | 0 ~ 1
        reward_rpm = (1 - self.rearRPM) + (1 - self.frontRPM)

        # 조건 7: 이동 거리
        reward_distance = self.state[0]

        # 조건 8: 가속도
        # 속도가 0 이상, VcruiseTarget 이하일 경우
        if 0 <= self.state[3] <= self.VcruiseTarget:
            # gForce가 0 ~ gForceTarget 사이일 경우
            if -0.2 <= self.gForce <= self.gForceTarget:
                reward_gForce = 1
            else:
                reward_gForce = -1
        
        # 속도가 VcruiseTarget 이상, VcruiseMax 이하일 경우
        elif self.VcruiseTarget < self.state[3] <= self.VcruiseMax:
            # gForce가 -0.1 ~ 0.1 사이일 경우
            if -0.2 <= self.gForce <= 0.2:
                reward_gForce = 1
            else:
                reward_gForce = -1

        # 속도가 VcruiseMax 이상일 경우
        elif self.state[3] > self.VcruiseMax:
            # gForce가 0 미만일 경우
            if self.gForce < 0:
                reward_gForce = 1
            else:
                reward_gForce = -1

        # 속도가 음수일 경우 속도를 높여야 함
        else:
            # gForce가 0 이하일 경우 가속도 상승
            if self.gForce > 0:
                reward_gForce = 1
            else:
                reward_gForce = -1

        # 조건 9: tilt 각 변화량
        # tilt 각이 0도로 변하면, tilt 각 변화량이 0이 되도록 유도
        if self.tilt_deg == 0:
            reward_tilt_delta = 1
        else:
            # tilt 각 변화량이 양수일 경우를 방지
            if self.tilt_deg_delta > 0:
                reward_tilt_delta = -1
            else:
                reward_tilt_delta = 1

        # 항공기 속도가 stallSpeed 이하일 경우와 이상일 경우, 가중치를 다르게 배정
        if self.state[3] < self.stallSpeed:
            weight = [500, 100, 20, 100, 100, 100, 2, 0, 0]
        else:
            weight = [500, 100,  1, 200, 200,  50, 1, 0, 0]

        rewards_list = [reward_tilt,     reward_pitch,    reward_time,
                        reward_speed,    reward_altitude, reward_rpm,
                        reward_distance, reward_gForce,   reward_tilt_delta]
        
        value_list = [self.tilt_deg, pitch_deg,     self.state[6],
                      self.state[3], self.state[1], [self.frontRPM, self.rearRPM],
                      self.state[0], self.gForce,   self.tilt_deg_delta]
        
        reward = np.dot(weight, rewards_list)

        return reward, rewards_list, value_list

    #################### step ####################
    def step(self, action):
        (self.frontRPM_delta, self.rearRPM_delta, self.elev_deg_delta, self.tilt_deg_delta) = action
        self.frontRPM +=   self.frontRPM_delta * self.rpm_rate
        self.rearRPM  +=   self.rearRPM_delta  * self.rpm_rate
        self.elev_deg +=   self.elev_deg_delta * self.elev_rate
        self.tilt_deg +=   self.tilt_deg_delta * self.tilt_rate

        self.frontRPM = np.clip(self.frontRPM, 0.0, 1.0)
        self.rearRPM  = np.clip(self.rearRPM, 0.0, 1.0)
        
        self.Simulation()
        
        reward, reward_detail, value_list = self.calculateReward()
        
        # Sharp reward(editing)
        done = False
        
        alt_constrain = self.altitudeDelta * 1.2
        
        if (np.abs(self.state[1])  >= alt_constrain) or \
           (self.pitchMin * 1.2 <= np.abs(math.degrees(self.state[2]))  <= self.pitchMax * 1.2) or \
           (self.tilt_deg < 0 or self.tilt_deg > 90) or \
           (self.state[3] > self.VcruiseMax):
            done = True
             
        observation = np.hstack((self.state[0],self.state[1],self.state[2],self.state[3],
                                 self.state[4],self.state[5],self.state[6],self.state[7],
                                 self.frontRPM, self.rearRPM, self.elev_deg, self.tilt_deg))

        info = {
            'reward_detail': reward_detail,
            'value_detail': value_list
        }
        return observation, reward, done, info
    
    def fqdot(self):
        return self.Myb/self.Iyy
    
    def fudot(self):
        return self.Fxb/self.m - self.state[5]*self.state[4]
    
    def fwdot(self):
        return self.Fzb/self.m + self.state[5]*self.state[3]
    
    def fthetadot(self):
        return self.state[5]
    
    def fxdot(self):
        return self.u*math.cos(self.state[2]) + self.w*math.sin(self.state[2])
    
    def fzdot(self):
        return -self.u*math.sin(self.state[2]) + self.w*math.cos(self.state[2])
    
    # Simulation
    def Simulation(self):
        self.x            = self.state[0] # m
        self.z            = self.state[1] # m
        self.theta        = self.state[2] # rad
        self.u            = self.state[3] # m/s
        self.w            = self.state[4] # m/s
        self.q            = self.state[5] # rad/s
        self.Sim_time     = self.state[6] # sec
        self.gForce       = self.state[7] # g
        
        # =============== Vehicle Model (Calculate Force&Moments) ===============
        self.frontThrust  = 2*self.K_T * (self.frontRPM * self.rpm_max)**2
        self.rearThrust   = 2*self.K_T * (self.rearRPM  * self.rpm_max)**2
        
        
        if self.state[3] == 0:
            self.aoa = self.state[2]
        else:
            self.aoa = math.atan(self.w/self.u)
        
        self.vel = math.sqrt(self.w**2 + self.u**2)
        
        aoa_deg = math.degrees(self.aoa)
        if aoa_deg < -15:
            CL_clean = linearInterpolation(-20, self.CL_table['-20'], -15, self.CL_table['-15'], aoa_deg)
            CD_clean = linearInterpolation(-20, self.CD_table['-20'], -15, self.CD_table['-15'], aoa_deg)
            Cm_clean = linearInterpolation(-20, self.Cm_table['-20'], -15, self.Cm_table['-15'], aoa_deg)
            
        elif aoa_deg <= 30:
            for i in range(-15, 35, 5):
                if aoa_deg <= i:
                    CL_clean = linearInterpolation(i-5, self.CL_table[str(i-5)], i, self.CL_table[str(i)], aoa_deg)
                    CD_clean = linearInterpolation(i-5, self.CD_table[str(i-5)], i, self.CD_table[str(i)], aoa_deg)
                    Cm_clean = linearInterpolation(i-5, self.Cm_table[str(i-5)], i, self.Cm_table[str(i)], aoa_deg)
                    break
        else:
            CL_clean = linearInterpolation(25, self.CL_table['25'], 30, self.CL_table['30'], aoa_deg)
            CD_clean = linearInterpolation(25, self.CD_table['25'], 30, self.CD_table['30'], aoa_deg)
            Cm_clean = linearInterpolation(25, self.Cm_table['25'], 30, self.Cm_table['30'], aoa_deg)

        CL_elev = self.elev_CL_0 + self.elev_CL_slop * self.elev_deg
        CD_elev = self.elev_CD_0 + self.elev_CD_slop * self.elev_deg
        Cm_elev = self.elev_Cm_0 + self.elev_Cm_slop * self.elev_deg
        
        if (aoa_deg >= -20) and (aoa_deg <= 30):
            self.CL = (CL_clean + CL_elev) # 항공기 전체 양력 계수입니다.
            self.CD = (CD_clean + CD_elev) # 항공기 전체 항력 계수입니다.

            self.L  = 0.5 * 1.225 * (self.vel**2) * self.S * (CL_clean + CL_elev)
            self.D  = 0.5 * 1.225 * (self.vel**2) * self.S * (CD_clean + CD_elev)
            self.Mp = 0.5 * 1.225 * (self.vel**2) * self.S * self.cbar * (Cm_clean + Cm_elev)
        else:
            self.L  = 0
            self.D  = 0
            self.Mp = 0
        
        self.Myb = self.cg2FrontProp_z*self.frontThrust*math.cos(math.radians(self.tilt_deg)) + self.cg2FrontProp_x*self.frontThrust*math.sin(math.radians(self.tilt_deg)) + self.cg2RearProp_x*self.rearThrust - self.cg2AeroCenter_z*(self.D*math.cos(self.aoa) + self.L*math.sin(self.aoa)) - self.cg2AeroCenter_x*(self.D*math.sin(self.aoa) - self.L*math.cos(self.aoa)) + self.Mp
        self.Fxb = self.frontThrust*math.cos(math.radians(self.tilt_deg)) - self.D*math.cos(self.aoa) - self.L*math.sin(self.aoa) + self.m*self.g*math.sin(self.aoa)
        self.Fzb = -self.rearThrust - self.frontThrust*math.sin(math.radians(self.tilt_deg)) + self.D*math.sin(self.aoa) - self.L*math.cos(self.aoa) + self.m*self.g*math.cos(self.aoa)
        # =============== Vehicle Model (Calculate Force&Moments) ===============
        
        # =============== Flight Dynamics with RK-4 (Calculate Next Status) ===============

        # q
        self.q += self.fqdot() * self.time_delta
        
        # u & ax
        self.u += self.fudot() * self.time_delta
        self.acceleration_x = self.fudot()
        
        # w & az
        self.w += self.fwdot() * self.time_delta
        self.acceleration_z = self.fwdot()
        
        # the
        self.theta += self.fthetadot() * self.time_delta
        
        # x
        self.x += self.fxdot() * self.time_delta
        
        # z
        self.z += self.fxdot() * self.time_delta

        self.Sim_time += self.time_delta

        # ckchoi: 가속도 계산 - 중력가속도 대비 비행기가 받는 가속도
        # self.gForce = math.sqrt(self.acceleration_x**2 + self.acceleration_z**2)/self.g
        self.gForce = self.acceleration_x / self.g
        
        # State Update
        self.state = [self.x, self.z, self.theta, self.u, self.w, self.q, self.Sim_time, self.gForce]
        # =============== Flight Dynamics with RK-4 (Calculate Next Status) ===============
    #################### step ####################
    
    #################### data collection ####################
    def dataCollection(self):
        data = {
            "time"     : self.Sim_time,
            "CL"       : self.CL,
            "CD"       : self.CD,
            "X"        : self.state[0],
            "Z"        : self.state[1],
            "theta"    : self.theta,
            "frontRPM" : self.frontRPM,
            "rearRPM"  : self.rearRPM,
            "U"        : self.u,
            "tilt_deg" : self.tilt_deg,
            "Lift"     : self.L,
            "L/W"      : round(self.L/(self.m*self.g), 8),
            "(L+T)/W"  : round((self.L + self.frontThrust + self.rearThrust)/(self.m*self.g), 8)
        }
        return data
    #################### data collection ####################

    #################### render ####################
    def render(self, mode='human'):
        self.draw_vehicle(self.state[0], self.state[1], self.state[2], math.radians(self.tilt_deg))
        
        texts = [
            ("Distance(m): ", round(self.state[0], 8)),
            ("Altitude(m): ", round(-self.state[1], 8)),
            ("Pitch(deg): ", round(math.degrees(self.state[2]), 8)),
            ("U(m/s): ", round(self.state[3], 8)),
            ("W(m/s): ", round(-self.state[4], 8)),
            ("Time(sec): ", round(self.state[6], 8)),
            ("Front RPM(%): ", round(self.frontRPM * 100, 8)),
            ("Rear RPM(%): ", round(self.rearRPM * 100, 8)),
            ("Elevator(deg): ", round(self.elev_deg, 8)),
            ("Tilt_Angle(deg): ", round(self.tilt_deg, 8)),
            ("Front Thrust(N): ", round(self.frontThrust, 8)),
            ("Rear Thrust(N): ", round(self.rearThrust, 8)),
            ("Weight(N): ", round(self.m * self.g, 8)),
            ("G(N): ", round(self.gForce, 8)),
            ("Lift(N): ", round(self.L, 8)),
            ("Drag(N): ", round(self.D, 8)),
            ("Lift/Weight: ", round(self.L / (self.m * self.g), 8)),
            ("Lift+Thrust/Weight: ", round((self.L + self.frontThrust + self.rearThrust) / (self.m * self.g), 8)),
            ("Pithching(N): ", round(self.Mp, 8)),
            ("AoA(deg): ", round(math.radians(self.aoa), 1))
        ]
        
        y = 10
        for text in texts:
            label = self.font.render(text[0] + str(text[1]), True, (28, 0, 0))
            self.screen.blit(label, (700, y))
            y += 20
        self.clock.tick(120)
        pygame.display.flip() 
    
    
    def draw_vehicle(self, x, z, theta, tilt_rad):
        rotated_image_vehicle = pygame.transform.rotate(self.vehicle,((math.degrees(theta))))
        rotated_image_Tilt_Prop = pygame.transform.rotate(self.Tilt_Prop, (math.degrees(theta + tilt_rad - math.pi/2)))
    
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
            
            self.screen.blit(rotated_image_Tilt_Prop, (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta) \
                                                       
                                             + (self.Tilt_Prop_height/2)*math.cos(vehicle_f_tilt_angle + theta + tilt_rad) - rotated_image_Tilt_Prop.get_width()/2,\
                                                 
                                             self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta) - (self.Tilt_Prop_height/2)*math.sin(vehicle_f_tilt_angle + theta + tilt_rad)\
                                                 
                                             - rotated_image_Tilt_Prop.get_height()/2))
            
            pygame.draw.line(self.screen, (179,179,179),\
                             
                             (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta)),\
                             
                             (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta) + 50*math.cos(theta + tilt_rad), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta) - 50*math.sin(theta + tilt_rad)), 4)
            
            pygame.draw.line(self.screen, (255,0,0),\
                             
                             (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta)),\
                             
                             (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta) + self.frontRPM*50*math.cos(theta + tilt_rad), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta) - self.frontRPM*50*math.sin(theta + tilt_rad)), 4)
            
            pygame.draw.line(self.screen, (179,179,179),\
                             
                             (200 + x - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta)),\
                             
                             (200 + x - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta) + 50*math.cos(theta + math.pi/2), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta) - 50*math.sin(theta + math.pi/2)), 4)
            
            pygame.draw.line(self.screen, (255,0,0),\
                             
                             (200 + x - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta)),\
                             
                             (200 + x - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta) + self.rearRPM*50*math.cos(theta + math.pi/2), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta) - self.rearRPM*50*math.sin(theta + math.pi/2)), 4)
                
            
        else:   
            self.screen.blit(rotated_image_vehicle, ((400 - rotated_image_vehicle.get_width()/2), (self.window_size[1]/2 + z - rotated_image_vehicle.get_height()/2)))
            
            self.screen.blit(rotated_image_Tilt_Prop, (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta)\
                                                       
                                             + (self.Tilt_Prop_height/2)*math.cos(vehicle_f_tilt_angle + theta + tilt_rad) - rotated_image_Tilt_Prop.get_width()/2,\
                                                 
                                             self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta) - (self.Tilt_Prop_height/2)*math.sin(vehicle_f_tilt_angle + theta + tilt_rad)\
                                                 
                                             - rotated_image_Tilt_Prop.get_height()/2))
            
            pygame.draw.line(self.screen, (179,179,179),\
                             
                             (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta)),\
                             
                             (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta) + 50*math.cos(theta + tilt_rad), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta) - 50*math.sin(theta + tilt_rad)), 4)
            
            pygame.draw.line(self.screen, (255,0,0),\
                             
                             (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta)),\
                             
                             (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta) + self.frontRPM*50*math.cos(theta + tilt_rad), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta) - self.frontRPM*50*math.sin(theta + tilt_rad)), 4)
            
            pygame.draw.line(self.screen, (179,179,179),\
                             
                             (400 - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta)),\
                             
                             (400 - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta) + 50*math.cos(theta + math.pi/2), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta) - 50*math.sin(theta + math.pi/2)), 4)
            
            pygame.draw.line(self.screen, (255,0,0),\
                             
                             (400 - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta)),\
                             
                             (400 - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta) + self.rearRPM*50*math.cos(theta + math.pi/2), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta) - self.rearRPM*50*math.sin(theta + math.pi/2)), 4)
                
    #################### render ####################

    #################### close ####################
    def close(self):
        if self.viewer != None:
            pygame.quit()
    ################### close ####################