import numpy as np
from scipy.ndimage import laplace
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 50
N = 100

w = L/N

# activator concentration
u = 1+0.01*np.random.rand(N,N)
Du = 1 # diffusion coeff of activator

# inhibitor concentration
v = 0.9+0.01*np.random.rand(N,N)
Dv = 9 # diffusion coeff of inhibitor

time = 600
dt = 0.2 * w**2 / max(Du, Dv)
timesteps = int(time/dt)

u_frames = []
v_frames = []

a = 0.1
b= 0.9

for _ in range(timesteps):
    lu = laplace(u, mode='reflect') / w**2
    lv = laplace(v, mode='reflect') / w**2

    uv2 = (u**2)*v
    u, v = (u+dt * (Du * lu + a - u + uv2), v+dt * (Dv*lv + b - uv2))

    if (_%1000 == 0):
        u_frames.append(u.copy())
        v_frames.append(v.copy())
    if (_%1000==0):
        print("frame",_,"/",timesteps)


fig, ax = plt.subplots(figsize=(5,5))
im = ax.imshow(v_frames[0], cmap='seismic', vmin=min(f.min() for f in v_frames),
               vmax=max(f.max() for f in v_frames))

def update(frame):
    im.set_data(v_frames[frame])
    #print(frame)
    return [im]

ani = FuncAnimation(
    fig,
    update,
    frames=len(v_frames),
    interval=50,
    blit=True,
    repeat=True
)
ani.save('turing.gif',fps=120)
plt.show()