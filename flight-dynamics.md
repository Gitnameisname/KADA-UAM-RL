# Flight Dynamics

## Flight Dynamics Equations
```python
def fqdot(self, q):
    return ((-self.f_Lz*math.cos(self.tilt_deg * math.pi/180) + self.f_Lx*math.sin(self.tilt_deg * math.pi/180))*self.T_f - self.r_Lx*self.T_r + self.Mp)/self.Iyy

def fudot(self, u):
    return -self.g*math.sin(self.state["theta"]) - self.state["q"]*self.state["W"] + (self.T_f*math.cos(self.tilt_deg * math.pi/180) - self.D*math.cos(self.aoa) - self.L*math.sin(self.aoa))/self.m

def fwdot(self, w):
    return self.g*math.cos(self.state["theta"]) + self.state["q"]*self.state["U"] + (- self.T_f*math.sin(self.tilt_deg * math.pi/180) - self.T_r + self.D*math.sin(self.aoa) - self.L*math.cos(self.aoa))/self.m
```