import gym
import control as ct
import numpy as np
#env = gym.make('CartPole-v1')
env = gym.make('Custom_CartPole-v0', thetaacc_error=0, initial_state_variation=[1,1,1,1])
g = env.gravity
M = env.masscart
m = env.masspole
l = env.length
Q = np.eye(4)*[10,1,1,1]
R = 1

'''System of equations'''
A = np.array([[0,1,0,0],[0,0,-m*g/M,0],[0,0,0,1],[0,0,(M+m)*g/(l*M),0]])
B = np.array([[0,1/M,0,-1/(l*M)]]).T


'''LQR'''
K,S,E = ct.lqr(A,B,Q,R)

'''Pole Placement'''
#K = ct.place(A,B,np.array([-1.1,-1.2,-1.3,-1.4]))


#env.x_threshold = 5.0
#env.theta_threshold_radians = 10.0




for i_episode in range(5):
    observation = env.reset()
    for t in range(500):
        env.render()
        #print(observation)
        #action = env.action_space.sample()
        u = -np.dot(K,observation)
        observation, reward, done, info = env.step(u[0])
        if done:
            print("Episode finished at time step {}".format(t+1))
            break
    print("Episode complete")
env.close()


# Shitty version
'''
import gym
import numpy as np

env = gym.make('CartPole-v0')

x = env.reset()
# x, xdot, theta, thetadot

gamma = (4.0 / 3.0 - env.masspole / env.masscart)

a = -env.gravity * env.masspole / (env.total_mass * gamma)
b = 1.0 / env.total_mass * (1 + env.masspole / (env.total_mass * gamma))
c = env.gravity / (env.length * gamma)
d = -1.0 / (env.total_mass * env.length * gamma)

tau = env.tau

F = np.array([
    [1, tau,       0,   0,       0],
    [0,   1, tau * a,   0, tau * b],
    [0,   0,       1, tau,       0],
    [0,   0, tau * c,   1, tau * d],
  ])

C = np.array([
    [0,  0, 0,  0,   0],
    [0, 10, 0,  0,   0],
    [0,  0, 100,  0,   0],
    [0,  0, 0, 0,   0],
    [0,  0, 0,  0, 1000],
  ])

c = np.array([0, 0, 0, 0, 0]).T

frame = 0
done = False
while not done:
  Ks = []
  T = 100
  V = np.zeros((4, 4))
  v = np.zeros((4))
  for t in range(T):
    t = T - t
    # Qt
    Qt = C + np.matmul(F.T, np.matmul(V, F))
    qt = c + np.matmul(F.T, v)

    Quu = Qt[-1:,-1:]
    Qux = Qt[-1:,:-1]
    Qxu = Qt[:-1, -1:]

    qu = qt[-1:]

    Qut_inv = np.linalg.inv(Quu)
    
    Kt = -np.matmul(Qut_inv, Qux)
    kt = -np.matmul(Qut_inv, qu)

    Ks.append((Kt, kt))
    
    V = Qt[:4, :4] + np.matmul(Qxu, Kt) + np.matmul(Qux, Kt.T) + np.matmul(Kt.T, np.matmul(Quu, Kt))
    v = qt[:4] + np.matmul(Qxu, kt) + np.matmul(Kt.T, Quu).reshape(-1) + np.matmul(Kt.T, np.matmul(Quu, kt))

  Kt, kt = Ks[-1]
  ut = np.matmul(Kt, x.reshape((1, -1)).T) + kt
  
  if ut > 0.0:
      ut = env.force_mag
      action = 1
  else:
      ut = -env.force_mag
      action = 0

  xu = np.hstack([x, ut])
  my_guess = np.matmul(F, xu.T)
  x, reward, done, info = env.step(action)
  frame += 1
  env.render()
print(frame)

'''
