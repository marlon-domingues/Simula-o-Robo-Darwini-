#TESTE TECNICA PID

import clases_Darwini as cla
import matplotlib.pyplot as plt 
import math

kp=0.1
ki=0.2
kd=0.1
total_cabo=[0]
setpoint_list=[]

PID=cla.PIDController(kp,ki,kd)

for i in range(1000):
    setpoint=math.sin(i/100)*10+i**0.5
    control_signal, error = PID.update(total_cabo[-1],setpoint)
    total_cabo.append(total_cabo[-1]+control_signal)
    setpoint_list.append(setpoint)

plt.plot(setpoint_list)
plt.plot(total_cabo)
plt.show()