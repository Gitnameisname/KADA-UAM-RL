import gym as gym
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
        self.setRender(window_size=[1000, 500])
        self.setSimulationData("aero.json")

        # Observation Space 설정
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(12,), dtype=np.float32)
        
        # 동작 공간 정의
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.set_init_state()
        self.seed()       
        
    def setSimulationData(self, DBname='aero.json'):

        SIMDB = dataLoader(DBname)
        CONSTDB = dataLoader("constrains.json")
        
        ## Simulation Data
        self.cg_x            = SIMDB["Configurations"]["cg_x"]                          # m
        self.cg_z            = SIMDB["Configurations"]["cg_z"]                          # m
        self.frontProp_x     = SIMDB["Configurations"]["frontProp_x"]                   # m
        self.frontProp_z     = SIMDB["Configurations"]["frontProp_z"]                   # m
        self.rearProp_x      = SIMDB["Configurations"]["rearProp_x"]                    # m
        self.rearProp_z      = SIMDB["Configurations"]["rearProp_z"]                    # m
        self.aeroCenter_x    = SIMDB["Configurations"]["aeroCenter_x"]                  # m
        self.aeroCenter_z        = SIMDB["Configurations"]["aeroCenter_z"]                  # m
        self.S               = SIMDB["Configurations"]["S"]                             # m^2
        self.cbar            = SIMDB["Configurations"]["cbar"]                          # m
        self.elevMaxDeg      = SIMDB["Configurations"]["elev_max"]                      # deg
        self.stallSpeed      = SIMDB["Configurations"]["stall_speed"]                    # m/s
        
        self.cg2FrontProp_x  = self.cg_x - self.frontProp_x                             # m
        self.cg2FrontProp_z  = self.cg_z - self.frontProp_z                             # m
        self.cg2RearProp_x   = self.cg_x - self.rearProp_x                              # m
        self.cg2RearProp_z   = self.cg_z - self.rearProp_z                              # m
        self.cg2AeroCenter_x = self.cg_x - self.aeroCenter_x                            # m
        self.cg2AeroCenter_z = self.cg_z - self.aeroCenter_z                                # m
        
        self.K_T             = SIMDB["Propulsion"]["K_T"]                               # none
        self.rpm_max         = SIMDB["Propulsion"]["rpm_max"]                           # rpm
        self.tilt_min        = SIMDB["Propulsion"]["tilt_min"]                          # rad
        self.tilt_max        = SIMDB["Propulsion"]["tilt_max"]                          # rad
        
        self.CL_table = {}
        self.CD_table = {}
        self.Cm_table = {}
        for aoa in range(-20, 31, 5):
            self.CL_table[str(aoa)] = SIMDB["Aerodynamics"]["CL"][str(aoa)]
            self.CD_table[str(aoa)] = SIMDB["Aerodynamics"]["CD"][str(aoa)]
            self.Cm_table[str(aoa)] = SIMDB["Aerodynamics"]["Cm"][str(aoa)]
        
        self.elev_CL_0    = SIMDB["Aerodynamics"]["elev"]["elev_CL_0"]                # none
        self.elev_CL_slop = SIMDB["Aerodynamics"]["elev"]["elev_CL_slop"]             # none/deg
        self.elev_CD_0    = SIMDB["Aerodynamics"]["elev"]["elev_CD_0"]                # none
        self.elev_CD_slop = SIMDB["Aerodynamics"]["elev"]["elev_CD_slop"]             # none/deg
        self.elev_Cm_0    = SIMDB["Aerodynamics"]["elev"]["elev_Cm_0"]                # none
        self.elev_Cm_slop = SIMDB["Aerodynamics"]["elev"]["elev_Cm_slop"]             # none/deg
        
        self.m          = SIMDB["WeightAndBalance"]["m"]                        # kg
        self.Iyy        = SIMDB["WeightAndBalance"]["Iyy"]                      # kg*m^2
        self.g          = SIMDB["WeightAndBalance"]["g"]                        # kg/m^2

        ## Constrains Data
        self.altitudeDelta = CONSTDB["Constrains"]["altitudeDelta"]                   # m
        self.VcruiseMax = CONSTDB["Constrains"]["VcruiseMax"]                         # m/s
        self.pitchMin   = CONSTDB["Constrains"]["pitchMin"]                           # deg
        self.pitchMax   = CONSTDB["Constrains"]["pitchMax"]                           # deg
        self.gForceMax  = CONSTDB["Constrains"]["g-forceMax"]                          # g

        ## Target Data
        self.VcruiseTarget = CONSTDB["Target"]["Vcruise"]                      # m/s
        self.pitchTarget   = CONSTDB["Target"]["pitch"]                        # deg
        self.gForceTarget = CONSTDB["Target"]["g-force"]                       # g
        
    def calcuateReward(self):
        # 복잡한 Reward를 더 자세히 알아보고 싶다면 /docs/Reward.md 파일을 참고하세요.

        # 조건 1: tilt각 차이 | tilt_deg가 0일 경우 최대(1) | tilt_deg가 작을 수록 좋음
        reward_tilt = (self.tilt_max - self.tilt_deg) / self.tilt_max

        # 조건 2: 피치 값 차이 | 피치가 pitchTarget일 경우 최대(1) | pitch가 pitchTarget에 가까울 수록 좋음
        pitch_deg = math.degrees(self.theta)
        pitch_width = self.pitchMax - self.pitchMin
        if self.pitchMin <= pitch_deg <= self.pitchTarget:
            slop = 2 * pitch_width / (self.pitchTarget - self.pitchMin)
            reward_pitch = (pitch_width - slop * np.abs(pitch_deg - self.pitchTarget)) / pitch_width
        elif self.pitchTarget <= pitch_deg <= self.pitchMax:
            slop = 2 * pitch_width / (self.pitchMax - self.pitchTarget)
            reward_pitch = (pitch_width - slop * np.abs(pitch_deg - self.pitchTarget)) / pitch_width
        else:
            reward_pitch = -1
        
        # 조건 3: 비행 시간 | 클수록 좋음 | 0 ~ inf, 1 timestep = 0.05 sec, 30,000 timestep = 1,500 sec = 25 min
        # 조건 3은 클수록 좋게 설정하였음(positive)
        reward_time = self.time
        
        # 조건 4: 크루즈 속도 차이 | speed가 VcruiseTarget일 경우 최대(1) | speed가 VcruiseTarget에 가까울 수록 좋음
        # 속도는 U와 W의 제곱근으로 계산
        speed = np.sqrt(self.u**2 + self.w**2)
        speed_width = self.VcruiseMax - self.VcruiseTarget
        if 0 <= speed <= self.VcruiseTarget:
            slop = 2 * speed_width / self.VcruiseTarget
            reward_speed = (speed_width - slop * np.abs(speed - self.VcruiseTarget)) / speed_width
        elif self.VcruiseTarget < speed <= self.VcruiseMax:
            slop = 2 * speed_width / (self.VcruiseMax - self.VcruiseTarget)
            reward_speed = (speed_width - slop * np.abs(speed - self.VcruiseTarget)) / speed_width
        else:
            reward_speed = -1

        # 조건 5: 순항 고도 | 고도가 0일 경우 최대(1) | 고도가 0에 가까울 수록 좋음
        reward_altitude = (self.altitudeDelta - np.abs(self.z)) / self.altitudeDelta

        # 조건 6: 프로펠러 rpm 최소화 | 0이 될 수록 좋음 | 0 ~ 1
        # 조건 6: 프로펠러 rpm 최소화 | 0이 될 수록 좋음 | 0 ~ 1
        reward_rpm = (1 - self.rearThrottle) + (1 - self.frontThrottle)

        # 조건 7: 이동 거리
        reward_distance = self.x

        # 조건 8: 가속도
        # 속도가 0 이상, VcruiseTarget 이하일 경우
        if 0 < speed <= self.VcruiseTarget:
            # gForce가 0 ~ gForceTarget 사이일 경우
            if 0 <= self.gForce <= self.gForceTarget:
                reward_gForce = 1
            # gForce가 음수이거나 gForceTarget를 초과할 경우
            else:
                reward_gForce = -1

        # 속도가 VcruiseTarget 이상, VcruiseMax 이하일 경우
        elif self.VcruiseTarget < speed <= self.VcruiseMax:
            # gForce가 -0.1 ~ 0.1 사이일 경우
            if -0.1 <= self.gForce <= 0.1:
                reward_gForce = 1
            else:
                reward_gForce = -1
        
        # 속도가 VcruiseMax 이상일 경우 속도를 줄여야 함
        elif speed > self.VcruiseMax:
            # gForce가 음수일 경우
            if self.gForce < 0:
                reward_gForce = 1
            else:
                reward_gForce = -1

        # 속도가 0 이하일 경우 속도를 높여야 함
        elif speed < 0:
            # gForce가 0 이하일 경우
            if self.gForce > 0:
                reward_gForce = 1
            else:
                reward_gForce = -1

        # 조건 9: tilt각 변화량
        # tilt각 변화량이 양수일 경우를 방지
        if self.tilt_delta > 0:
            reward_tilt_delta = -1
        else:
            reward_tilt_delta = 1

        # 항공기 속도가 stallSpeed 이하일 경우와 이상일 경우, 가중치를 다르게 배정
        if speed < self.stallSpeed:
            weight = [500, 100, 20,
                      100, 100, 100,
                      2, 10, 10]
        else:
            weight = [500, 100,  1,
                      200, 200,  50,
                      1, 1, 100]

        rewards_list = [reward_tilt, reward_pitch, reward_time,
                        reward_speed, reward_altitude, reward_rpm,
                        reward_distance, reward_gForce, reward_tilt_delta]

        value_list = [self.tilt_deg, pitch_deg, self.time,
                        speed, self.z, [self.frontThrottle, self.rearThrottle],
                        self.x, self.gForce, self.tilt_delta]
        reward = np.dot(weight, rewards_list)

        return reward, rewards_list, value_list
    
    def set_init_state(self):
        self.state = [0] * 12

        self.x = 0
        self.z = 0
        self.theta = 0
        self.u = 0
        self.w = 0
        self.q = 0
        self.time = 0
                
        # Actor의 Action 값들
        # 초기 네 개의 로터 스로틀은 호버링 상태로 설정
        self.frontThrottle = round(math.sqrt(self.m*self.g/(4*self.K_T))/self.rpm_max, 3)
        self.rearThrottle  = round(math.sqrt(self.m*self.g/(4*self.K_T))/self.rpm_max, 3)
        self.elev_deg = 0
        self.tilt_deg = 90

        self.time_delta = 0.05   
        self.rpm_rate = 0.01             # ratio (rpm/rpm_max)
        self.elev_rate = 1               # deg
        self.tilt_rate = 1               # deg

        self.current_score = 0
        self.viewer = None       
        
        self.frontThrust = 0
        self.rearThrust = 0
        self.L = 0
        self.D = 0
        self.Mp = 0
        self.aoa = 0
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
    
    def set_observation(self):
        observation = np.hstack([self.x, self.z, self.theta,
                                self.u, self.w, self.q,
                                self.time, self.gForce,
                                self.frontThrottle, self.rearThrottle, self.elev_deg, self.tilt_deg])

        return observation

    #################### reset ####################
    def reset(self):
        self.set_init_state()
        self.current_score = 0

        return self.set_observation()
    #################### reset ####################    
    
    def close(self):
        if self.viewer is not None:
            pygame.quit()
            self.viewer = None

    #################### step ####################
    def step(self, action):
        (self.frontThrottle_delta, self.rearThrottle_delta, self.elev_delta, self.tilt_delta) = action
        
        self.frontThrottle += self.frontThrottle_delta * self.rpm_rate
        self.rearThrottle  += self.rearThrottle_delta * self.rpm_rate
        self.elev_deg      += self.elev_delta * self.elev_rate
        self.tilt_deg      += self.tilt_delta * self.tilt_rate

        # np.clip(값, 최소값, 최대값) : 값이 최소값보다 작으면 최소값으로, 최대값보다 크면 최대값으로 설정
        # frontThrottle와 rearThrottle의 값이 0보다 작으면 0으로, 1보다 크면 1로 설정
        self.frontThrottle = np.clip(self.frontThrottle, 0.0, 1.0)
        self.rearThrottle = np.clip(self.rearThrottle, 0.0, 1.0)
        
        self.Simulation()
        
        reward, rewards_list, value_list = self.calcuateReward()
        
        done = False
        # 고도가 altitudeDelta 이상이거나 이하일 경우, 피치가 pitchMin 이상 pitchMax 이하일 경우, 비행 속도가 VcruiseMax 이상일 경우 종료
        if np.abs(self.z) >= self.altitudeDelta or \
           not (self.pitchMin <= self.theta <= self.pitchMax) or \
           not (self.tilt_min <= self.tilt_deg <= self.tilt_max) or \
           self.u > self.VcruiseMax:
            done = True
             
        observation = self.set_observation()
        step_data = self.dataCollection()
        info = {
            'Time' : self.time,
            'x_pos': self.x,
            'z_pos': self.z,
            'pitch': self.theta,
            'reward_detail': rewards_list,
            'value_detail': value_list,
            'data': step_data
        }
        return observation, reward, done, info
    
    def fqdot(self):
        return self.Myb/self.Iyy
    
    def fudot(self):
        return self.Fxb/self.m + self.q*self.w
    
    def fwdot(self):
        return self.Fzb/self.m + self.q*self.u
    
    def fthetadot(self):
        return self.q
    
    def fxdot(self):
        return self.u*math.cos(self.theta) + self.w*math.sin(self.theta)
    
    def fzdot(self):
        return -self.u*math.sin(self.theta) + self.w*math.cos(self.theta)
    
    # Simulation
    def Simulation(self):
        self.x            = self.state[0]
        self.z            = self.state[1]
        self.theta        = self.state[2]
        self.u            = self.state[3]
        self.w            = self.state[4]
        self.q            = self.state[5]
        self.time         = self.state[6]
        self.gForce       = self.state[7]
        
        # =============== Vehicle Model (Calculate Force&Moments) ===============
        self.frontThrust = 2 * self.K_T * (self.frontThrottle * self.rpm_max)**2
        self.rearThrust  = 2 * self.K_T * (self.rearThrottle  * self.rpm_max)**2
        
        if self.theta == 0:
            self.aoa = self.theta
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
        
        if -20 <= aoa_deg <= 30:
            self.CL = CL_clean + CL_elev  # 항공기 전체 양력 계수입니다.
            self.CD = CD_clean + CD_elev  # 항공기 전체 항력 계수입니다.
            self.Cm = Cm_clean + Cm_elev  # 항공기 전체 피치 모멘트 계수입니다.

            self.L = 0.5 * 1.225 * self.vel**2 * self.S * (self.CL)  # 항공기 전체 양력입니다.
            self.D = 0.5 * 1.225 * self.vel**2 * self.S * (self.CD)  # 항공기 전체 항력입니다.
            self.Mp = 0.5 * 1.225 * self.vel**2 * self.S * self.cbar * (self.Cm)  # 항공기 전체 피치 모멘트입니다.
        else:
            self.L = 0
            self.D = 0
            self.Mp = 0
        
        tilt_radians = math.radians(self.tilt_deg)
        self.Myb = self.cg2FrontProp_z * self.frontThrust * math.cos(tilt_radians) + self.cg2FrontProp_x * self.frontThrust * math.sin(tilt_radians) + self.cg2RearProp_x * self.rearThrust - self.cg2AeroCenter_z * (self.D * math.cos(math.radians(self.aoa)) + self.L * math.sin(math.radians(self.aoa))) - self.cg2AeroCenter_x * (self.D * math.sin(math.radians(self.aoa)) - self.L * math.cos(math.radians(self.aoa))) + self.Mp
        self.Fxb = self.frontThrust * math.cos(tilt_radians) - self.D * math.cos(math.radians(self.aoa)) - self.L * math.sin(math.radians(self.aoa)) + self.m * self.g * math.sin(math.radians(self.aoa))
        self.Fzb = -self.rearThrust - self.frontThrust * math.sin(tilt_radians) + self.D * math.sin(math.radians(self.aoa)) - self.L * math.cos(math.radians(self.aoa)) + self.m * self.g * math.cos(math.radians(self.aoa))
        # =============== Vehicle Model (Calculate Force&Moments) ===============
        
        # =============== Flight Dynamics with RK-4 (Calculate Next Status) ===============
        self.q += self.time_delta * self.fqdot()
        self.u += self.time_delta * self.fudot()
        self.w += self.time_delta * self.fwdot()
        self.theta += self.time_delta * self.fthetadot()
        self.x += self.time_delta * self.fxdot()
        self.z += self.time_delta * self.fzdot()
        # =============== Flight Dynamics with RK-4 (Calculate Next Status) ===============

        # 가속도 계산
        self.acceleration = math.sqrt(self.fudot()**2 + self.fwdot()**2)
        self.gForce = self.acceleration / self.g

        # 시간 증가
        self.time += self.time_delta
        
        # State Update
        self.state = [self.x, self.z, self.theta, self.u, self.w, self.q, self.time, self.gForce, self.frontThrottle, self.rearThrottle, self.elev_deg, self.tilt_deg]
    #################### step ####################
    
    #################### data collection ####################
    def dataCollection(self):
        data = {
            "time": self.time,
            "CL": self.CL,
            "CD": self.CD,
            "X": self.x,
            "Z": self.z,
            "theta": self.theta,
            "frontThrottle": self.frontThrottle,
            "rearThrottle": self.rearThrottle,
            "U": self.u,
            "tilt": self.tilt_deg,
            "Lift": self.L,
            "L/W": round(self.L/(self.m*self.g), 8),
            "(L+T)/W": round((self.L + self.frontThrust + self.rearThrust)/(self.m*self.g), 8)
        }
        return data
    #################### data collection ####################

    #################### render ####################
    def setRender(self, window_size):
        self.window_size = window_size
        img_path = "src/image"

        self.background = pygame.image.load(os.path.join(f"{img_path}/Background.png"))
        self.vehicle = pygame.image.load(os.path.join(f"{img_path}/vehicle.png"))
        vehicle_size = self.vehicle.get_rect().size
        self.vehicle_width = vehicle_size[0]
        self.vehicle_height = vehicle_size[1]

        self.Tilt_Prop = pygame.image.load(os.path.join(f"{img_path}/Tiltprop.png"))
        Tilt_Prop_size = self.Tilt_Prop.get_rect().size
        self.Tilt_Prop_width = Tilt_Prop_size[0]
        self.Tilt_Prop_height = Tilt_Prop_size[1]
        self.Textboard = pygame.image.load(os.path.join(f"{img_path}/textboard(300x200).png"))
        
        pygame.font.init()
        self.font = pygame.font.SysFont('arial',20, True, True)  #폰트 설정
        
    def render(self, mode='human'):
        self.drawVehicle(self.x, self.z, self.theta, self.tilt_deg)

        state_texts = {
            "Distance(m)": round(self.x, 8),
            "Altitude(m)": round(self.z, 8),
            "Pitch(deg)": round(math.degrees(self.theta), 8),
            "U(m/s)": round(self.u, 8),
            "W(m/s)": round(self.w, 8),
            "Time(sec)": round(self.time, 8),
            "Front_RPM(%)": round(self.frontThrottle*100, 8),
            "Rear_RPM(%)": round(self.rearThrottle*100, 8),
            "Elevator(deg)": round(self.elev_deg, 8),
            "Tilt_Angle(deg)": round(self.tilt_deg, 8),
            "Front Thrust(N)": round(self.frontThrust, 8),
            "Rear Thrust(N)": round(self.rearThrust, 8),
            "Weight(N)": round(self.m*self.g, 8),
            "G(N)": round(self.gForce, 8),
            "Lift(N)": round(self.L, 8),
            "Drag(N)": round(self.D, 8),
            "Lift/Weight": round(self.L/(self.m*self.g), 8),
            "Lift+Thrust/Weight": round((self.L + self.frontThrust + self.rearThrust)/(self.m*self.g), 8),
            "Pitching(N)": round(self.Mp, 8),
            "AoA(deg)": round(self.aoa, 1)
        }

        for i, (k, v) in enumerate(state_texts.items()):
            text = self.font.render(f"{k}: {v}", True, (28, 0, 0))
            self.screen.blit(text, (700, 10 + 20*i))
        
        self.clock.tick(120)
        pygame.display.flip()
    
    def drawVehicle(self, x, z, theta, tilt):
        theta = math.radians(theta)
        radians_90 = math.radians(90)

        rotated_image_vehicle = pygame.transform.rotate(self.vehicle,math.degrees(theta))
        rotated_image_Tilt_Prop = pygame.transform.rotate(self.Tilt_Prop, (math.degrees(theta + tilt - radians_90)))   
        
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
                                                       
                                             + (self.Tilt_Prop_height/2)*math.cos(vehicle_f_tilt_angle + theta + tilt) - rotated_image_Tilt_Prop.get_width()/2,\
                                                 
                                             self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta) - (self.Tilt_Prop_height/2)*math.sin(vehicle_f_tilt_angle + theta + tilt)\
                                                 
                                             - rotated_image_Tilt_Prop.get_height()/2))
            
            pygame.draw.line(self.screen, (179,179,179),\
                             
                             (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta)),\
                             
                             (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta) + 50*math.cos(theta + tilt), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta) - 50*math.sin(theta + tilt)), 4)
            
            pygame.draw.line(self.screen, (255,0,0),\
                             
                             (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta)),\
                             
                             (200 + x + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta) + self.frontThrottle*50*math.cos(theta + tilt), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta) - self.frontThrottle*50*math.sin(theta + tilt)), 4)
            
            pygame.draw.line(self.screen, (179,179,179),\
                             
                             (200 + x - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta)),\
                             
                             (200 + x - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta) + 50*math.cos(theta + radians_90), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta) - 50*math.sin(theta + radians_90)), 4)
            
            pygame.draw.line(self.screen, (255,0,0),\
                             
                             (200 + x - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta)),\
                             
                             (200 + x - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta) + self.rearThrottle*50*math.cos(theta + radians_90), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta) - self.rearThrottle*50*math.sin(theta + radians_90)), 4)
                
        else:   
            self.screen.blit(rotated_image_vehicle, ((400 - rotated_image_vehicle.get_width()/2), (self.window_size[1]/2 + z - rotated_image_vehicle.get_height()/2)))
            
            self.screen.blit(rotated_image_Tilt_Prop, (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta)\
                                                       
                                             + (self.Tilt_Prop_height/2)*math.cos(vehicle_f_tilt_angle + theta + tilt) - rotated_image_Tilt_Prop.get_width()/2,\
                                                 
                                             self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta) - (self.Tilt_Prop_height/2)*math.sin(vehicle_f_tilt_angle + theta + tilt)\
                                                 
                                             - rotated_image_Tilt_Prop.get_height()/2))
            
            pygame.draw.line(self.screen, (179,179,179),\
                             
                             (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta)),\
                             
                             (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta) + 50*math.cos(theta + tilt), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta) - 50*math.sin(theta + tilt)), 4)
            
            pygame.draw.line(self.screen, (255,0,0),\
                             
                             (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta)),\
                             
                             (400 + f_tilt_rotate_pos_length*math.cos(vehicle_f_tilt_angle + theta) + self.frontThrottle*50*math.cos(theta + tilt), self.window_size[1]/2 + z - f_tilt_rotate_pos_length*math.sin(vehicle_f_tilt_angle + theta) - self.frontThrottle*50*math.sin(theta + tilt)), 4)
            
            pygame.draw.line(self.screen, (179,179,179),\
                             
                             (400 - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta)),\
                             
                             (400 - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta) + 50*math.cos(theta + radians_90), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta) - 50*math.sin(theta + radians_90)), 4)
            
            pygame.draw.line(self.screen, (255,0,0),\
                             
                             (400 - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta)),\
                             
                             (400 - r_tilt_rotate_pos_length*math.cos(vehicle_r_tilt_angle + theta) + self.rearThrottle*50*math.cos(theta + radians_90), self.window_size[1]/2 + z + r_tilt_rotate_pos_length*math.sin(vehicle_r_tilt_angle + theta) - self.rearThrottle*50*math.sin(theta + radians_90)), 4)
    #################### render ####################