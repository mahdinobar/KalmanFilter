# -*- coding: utf-8 -*-
"""
Kalman filter implementation in Python and C++
Mahdi Nobar
mnobar@ethz.ch

"""
import numpy as np
import matplotlib.pyplot as plt
from myKalmanFilter import KalmanFilter

# initial values for the simulation
initialPosition = np.array([514.5, -269.8, 67.6])  # [mm]
initialVelocity = np.array([0, 3 * 10 / 1000, 0])  # [mm/1ms]



# number of discretization time steps
N = 28232  # dt=1[ms]

# define the system matrices - Newtonian system
# system matrices and covariances
A = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
B = np.matrix([[1]])
C = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# measurement noise covariance
R = np.array([[2e-3 ** 2, 0, 0], [0, 2e-3 ** 2, 0], [0, 0, 2e-3 ** 2]])
# process uncertainty covariance
Q = np.array([[1e-3 ** 2, 0, 0], [0, 1e-3 ** 2, 0], [0, 0, 1e-3 ** 2]]) # np.matrix(np.zeros((3, 3)))

# guess of the initial estimate
x0 = np.array([510, -265, 65])
# initial covariance matrix 
P0 = 25 * np.matrix(np.eye(3))

# time vector for simulation
tVec = np.linspace(0, (N - 1), N)

# IDEAL STATES
# vector used to store the simulated X_true
X_true = np.zeros((3, np.size(tVec)))
velocity = np.zeros((3, np.size(tVec)))
acceleration = np.zeros((3, np.size(tVec)))
# simulate the system behavior (discretize system)
for k in np.arange(np.size(tVec)):
    for i in range(0, 3):
        X_true[i, k] = initialPosition[i] + initialVelocity[i] * tVec[k]
        velocity[i, k] = initialVelocity[i]

# MEASURED(observed) STATES
# add the measurement noise
X_measured = np.zeros((3, np.size(tVec)))
for i in range(0, 3):
    X_measured[i, :] = X_true[i, :] + np.random.normal(0, np.sqrt(R[i,i]), size=np.size(tVec))

# here, we save the noisy outputs such that we can use them in C++ code
np.savetxt('myOutput_x.csv', X_measured[0, :], delimiter=',')

# verify the X_true vector by plotting the results

plotStep = N // 100
fig, ax = plt.subplots(3, 1, figsize=(12, 8))
ax[0].plot(tVec[0:plotStep], X_true[0, 0:plotStep], '-*g', linewidth=1, label='ideal')
ax[0].plot(tVec[0:plotStep], X_measured[0, 0:plotStep], '-or', linewidth=1, label='observed')
ax[0].set_ylabel("x [mm]", fontsize=14)
ax[0].legend()
ax[1].plot(tVec[0:plotStep], X_true[1, 0:plotStep], '-*g', linewidth=1, label='ideal')
ax[1].plot(tVec[0:plotStep], X_measured[1, 0:plotStep], '-or', linewidth=1, label='observed')
ax[1].set_ylabel("y [mm]", fontsize=14)
ax[1].legend()
ax[2].plot(tVec[0:plotStep], X_true[2, 0:plotStep], '-*g', linewidth=1, label='ideal')
ax[2].plot(tVec[0:plotStep], X_measured[2, 0:plotStep], '-or', linewidth=1, label='observed')
ax[2].set_xlabel("$t_{k}$ [ms]", fontsize=14)
ax[2].set_ylabel("z [mm]", fontsize=14)
ax[2].legend()
plt.tight_layout()
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
u = initialVelocity[1]  # u(t)=speed_y=v_y=ctd
# simulate online prediction
for j in np.arange(np.size(tVec)):
    print(j)
    KalmanFilterObject.propagateDynamics(u)
    KalmanFilterObject.computeAposterioriEstimate(X_measured[:, j])

KalmanFilterObject.estimates_aposteriori

# extract the state estimates in order to plot the results
x_hat = []
y_hat = []
z_hat = []
# Here are the C++ estimates
x_hat_cpp = []
y_hat_cpp = []
z_hat_cpp = []
# Load C++ estimates
cppEstimates = np.loadtxt("myEstimatesAposteriori.csv", delimiter=",")
for j in np.arange(np.size(tVec)):
    # python estimates
    x_hat.append(KalmanFilterObject.estimates_aposteriori[0, j])
    y_hat.append(KalmanFilterObject.estimates_aposteriori[1, j])
    z_hat.append(KalmanFilterObject.estimates_aposteriori[2, j])
    # # cpp estimates
    # x_hat_cpp.append(cppEstimates[0, j])
    # y_hat_cpp.append(cppEstimates[1, j])
    # z_hat_cpp.append(cppEstimates[2, j])

plotStep = N // 100
k0=1
fig, ax = plt.subplots(3, 1, figsize=(12, 8))
ax[0].plot(tVec[k0:plotStep], X_true[0, k0:plotStep], '-*g', linewidth=1, label='ideal')
ax[0].plot(tVec[k0:plotStep], X_measured[0, k0:plotStep], '-or', linewidth=1, label='measured')
ax[0].plot(tVec[k0:plotStep], x_hat[k0:plotStep], '-ob', linewidth=1, label='estimated')
ax[0].set_ylabel("x [mm]", fontsize=14)
ax[0].legend()
ax[1].plot(tVec[k0:plotStep], X_true[1, k0:plotStep], '-*g', linewidth=1, label='ideal')
ax[1].plot(tVec[k0:plotStep], X_measured[1, k0:plotStep], '-or', linewidth=1, label='measured')
ax[1].plot(tVec[k0:plotStep], y_hat[k0:plotStep], '-ob', linewidth=1, label='estimated')
ax[1].set_ylabel("y [mm]", fontsize=14)
ax[1].legend()
ax[2].plot(tVec[k0:plotStep], X_true[2, k0:plotStep], '-*g', linewidth=1, label='ideal')
ax[2].plot(tVec[k0:plotStep], X_measured[2, k0:plotStep], '-or', linewidth=1, label='measured')
ax[2].plot(tVec[k0:plotStep], z_hat[k0:plotStep], '-ob', linewidth=1, label='estimated')
ax[2].set_xlabel("$t_{k}$ [ms]", fontsize=14)
ax[2].set_ylabel("z [mm]", fontsize=14)
ax[2].legend()
plt.tight_layout()
fig.savefig('data.png', dpi=600)
plt.show()

# # plot the difference between the Python and C++ estimates
# errorEstimators1 = np.array(x_hat) - np.array(x_hat_cpp)
# errorEstimators2 = np.array(y_hat) - np.array(y_hat_cpp)
# errorEstimators3 = np.array(z_hat) - np.array(z_hat_cpp)
# fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
# ax2.plot(steps, errorEstimators1, color='red', linestyle='-', linewidth=2, label='(python_est-cpp_est)_{state 1}')
# ax2.plot(steps, errorEstimators2, color='blue', linestyle='-', linewidth=2, label='(python_est-cpp_est)_{state 2}')
# ax2.plot(steps, errorEstimators3, color='magenta', linestyle='-', linewidth=2, label='(python_est-cpp_est)_{state 3}')
# ax2.set_xlabel("Discrete-time steps k", fontsize=14)
# ax2.set_ylabel("Difference between implementations", fontsize=14)
# ax2.tick_params(axis='both', labelsize=12)
# ax2.grid()
# ax2.legend(fontsize=14)
# fig2.savefig('estimationErrorsImplementation.png', dpi=600)
# plt.show()
print("end")
