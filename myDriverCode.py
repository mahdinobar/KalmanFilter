# -*- coding: utf-8 -*-
"""
Kalman filter implementation in Python and C++
Mahdi Nobar
mnobar@ethz.ch

"""
###############################################################################
# RUN THIS PART FIRST, FROM HERE 
# UNTIL - SCROLL DOWN
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from myKalmanFilter import KalmanFilter

# discretization step
h = 1
# initial values for the simulation
initialPosition = np.array([514.5, -269.8, 67.6])  # [mm]
initialVelocity = np.array([0, 3 * 10 / 1000, 0])  # [mm/1ms]

# measurement noise standard deviation
noiseStd = np.array([1e-3, 1e-3, 1e-3]);
# number of discretization time steps
numberTimeSteps = 100  # dt=1[ms]

# define the system matrices - Newtonian system
# system matrices and covariances
A = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
B = np.matrix([[1]])
C = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

R = 1*np.matrix(np.eye((3)))  # (noiseStd**2)*np.matrix([[1]])
Q = np.matrix(np.zeros((3, 3)))

# guess of the initial estimate
x0 = np.array([[0], [0], [0]])
# initial covariance matrix 
P0 = 1 * np.matrix(np.eye(3))

# time vector for simulation
timeVector = np.linspace(0, (numberTimeSteps - 1) * h, numberTimeSteps)

# IDEAL STATES
# vector used to store the simulated position
position = np.zeros((3, np.size(timeVector)))
velocity = np.zeros((3, np.size(timeVector)))
acceleration = np.zeros((3, np.size(timeVector)))
# simulate the system behavior (discretize system)
for k in np.arange(np.size(timeVector)):
    for i in range(0, 3):
        position[i, k] = initialPosition[i] + initialVelocity[i] * timeVector[k]
        velocity[i, k] = initialVelocity[i]

# MEASURED(observed) STATES
# add the measurement noise
positionNoisy = np.zeros((3, np.size(timeVector)))
for i in range(0, 3):
    positionNoisy[i, :] = position[i, :] + np.random.normal(0, noiseStd[i], size=np.size(timeVector))

# here, we save the noisy outputs such that we can use them in C++ code
np.savetxt('myOutput_x.csv', positionNoisy[0,:], delimiter=',')

# verify the position vector by plotting the results

plotStep = numberTimeSteps
fig, ax = plt.subplots(3, 1, figsize=(10, 15))
ax[0].plot(timeVector[0:plotStep], position[0, 0:plotStep], '-*g', linewidth=2, label='ideal')
ax[0].plot(timeVector[0:plotStep], positionNoisy[0, 0:plotStep], '-or', label='observed')
ax[0].set_xlabel("time-step k", fontsize=14)
ax[0].set_ylabel("x [mm]", fontsize=14)
ax[0].legend()
ax[1].plot(timeVector[0:plotStep], position[1, 0:plotStep], '-*g', linewidth=2, label='ideal')
ax[1].plot(timeVector[0:plotStep], positionNoisy[1, 0:plotStep], '-or', label='observed')
ax[1].set_xlabel("time-step steps k", fontsize=14)
ax[1].set_ylabel("y [mm]", fontsize=14)
ax[1].legend()
ax[2].plot(timeVector[0:plotStep], position[2, 0:plotStep], '-*g',linewidth=2, label='ideal')
ax[2].plot(timeVector[0:plotStep], positionNoisy[2, 0:plotStep], '-or', label='observed')
ax[2].set_xlabel("time-step steps k", fontsize=14)
ax[2].set_ylabel("z [mm]", fontsize=14)
ax[2].legend()
fig.savefig('data.png', dpi=600)
plt.show()

###############################################################################
# BEFORE RUNNING THE REMAINING PART OF THE CODE
# RUN THE C++ CODE THAT IMPLEMENTS THE KALMAN FILTER 
# THE C++ CODE 
# AFTER THAT, YOU CAN RUN THE CODE GIVEN BELOW
###############################################################################

# create a Kalman filter object
KalmanFilterObject = KalmanFilter(x0, P0, A, B, C, Q, R)
inputValue = initialVelocity[1] #u(t)=speed_y=v_y=ctd
# simulate online prediction
for j in np.arange(np.size(timeVector)):
    print(j)
    KalmanFilterObject.propagateDynamics(inputValue)
    KalmanFilterObject.computeAposterioriEstimate(positionNoisy[:,j])

KalmanFilterObject.estimates_aposteriori

# extract the state estimates in order to plot the results
estimate1 = []
estimate2 = []
estimate3 = []
# Here are the C++ estimates
estimate1cpp = []
estimate2cpp = []
estimate3cpp = []
# Load C++ estimates
cppEstimates = np.loadtxt("myEstimatesAposteriori.csv", delimiter=",")
for j in np.arange(np.size(timeVector)):
    # python estimates
    estimate1.append(KalmanFilterObject.estimates_aposteriori[0,j])
    estimate2.append(KalmanFilterObject.estimates_aposteriori[1,j])
    estimate3.append(KalmanFilterObject.estimates_aposteriori[2,j])
    # cpp estimates
    estimate1cpp.append(cppEstimates[0, j])
    estimate2cpp.append(cppEstimates[1, j])
    estimate3cpp.append(cppEstimates[2, j])

# create vectors corresponding to the true values in order to plot the results
estimate1true = position
estimate2true = velocity
estimate3true = acceleration

# plot the results
steps = np.arange(np.size(timeVector))
fig, ax = plt.subplots(3, 1, figsize=(10, 15))
ax[0].plot(steps, estimate1true[0,:], color='red', linestyle='-', linewidth=8, label='True x')
ax[0].plot(steps, estimate1, color='blue', linestyle='-', linewidth=7, label='Estimate of x -Python')
ax[0].plot(steps, estimate1cpp, color='magenta', linestyle='-', linewidth=2, label='Estimate of x -C++')
ax[0].set_xlabel("Discrete-time steps k", fontsize=14)
ax[0].set_ylabel("x", fontsize=14)
ax[0].tick_params(axis='both', labelsize=12)
# ax[0].set_yscale('log')
# ax[0].set_ylim(98,102)
ax[0].grid()
ax[0].legend(fontsize=14)

ax[1].plot(steps, estimate2true[1,:], color='red', linestyle='-', linewidth=8, label='True y')
ax[1].plot(steps, estimate2, color='blue', linestyle='-', linewidth=7, label='Estimate of y - Python')
ax[1].plot(steps, estimate2cpp, color='magenta', linestyle='-', linewidth=2, label='Estimate of y - C++')
ax[1].set_xlabel("Discrete-time steps k", fontsize=14)
ax[1].set_ylabel("y", fontsize=14)
ax[1].tick_params(axis='both', labelsize=12)
# ax[0].set_yscale('log')
# ax[1].set_ylim(0,2)
ax[1].grid()
ax[1].legend(fontsize=14)

ax[2].plot(steps, estimate3true[2,:], color='red', linestyle='-', linewidth=8, label='True z')
ax[2].plot(steps, estimate3, color='blue', linestyle='-', linewidth=7, label='Estimate of z - Python')
ax[2].plot(steps, estimate3cpp, color='magenta', linestyle='-', linewidth=2, label='Estimate of z - C++')
ax[2].set_xlabel("Discrete-time steps k", fontsize=14)
ax[2].set_ylabel("z", fontsize=14)
ax[2].tick_params(axis='both', labelsize=12)
# ax[0].set_yscale('log')
# ax[1].set_ylim(0,2)
ax[2].grid()
ax[2].legend(fontsize=14)
fig.savefig('plots.png', dpi=600)
plt.show()

# plot the difference between the Python and C++ estimates
errorEstimators1 = np.array(estimate1) - np.array(estimate1cpp)
errorEstimators2 = np.array(estimate2) - np.array(estimate2cpp)
errorEstimators3 = np.array(estimate3) - np.array(estimate3cpp)
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
ax2.plot(steps, errorEstimators1, color='red', linestyle='-', linewidth=2, label='(python_est-cpp_est)_{state 1}')
ax2.plot(steps, errorEstimators2, color='blue', linestyle='-', linewidth=2, label='(python_est-cpp_est)_{state 2}')
ax2.plot(steps, errorEstimators3, color='magenta', linestyle='-', linewidth=2, label='(python_est-cpp_est)_{state 3}')
ax2.set_xlabel("Discrete-time steps k", fontsize=14)
ax2.set_ylabel("Difference between implementations", fontsize=14)
ax2.tick_params(axis='both', labelsize=12)
ax2.grid()
ax2.legend(fontsize=14)
fig2.savefig('estimationErrorsImplementation.png', dpi=600)
plt.show()
