# -*- coding: utf-8 -*-
"""
Kalman filter implementation in Python and C++
Mahdi Nobar
mnobar@ethz.ch

"""
import numpy as np
import matplotlib.pyplot as plt
from myKalmanFilter import KalmanFilter
import cv2


def main_model_0(log_dir):
    # initial values for the simulation
    # x_hat_init = np.array([514.5, -269.8, 67.6])  # [mm]
    x_hat_init = np.array(
        [9.5 * 50 + 30 + 5 + 4, -5.5 * 50 + 5, 90 + 3 - 25 + 33])  # [mm] manually measured and fixed rigidly
    # v_hat_init = np.array([0, 3 * 10 / 1000, 0])  # [mm/1ms]
    v_hat_init = np.array([0, 0.0341, 0])  # [mm/1ms]

    # number of discretization time steps
    N = 9066  # dt=1[ms]

    # define the system matrices - Newtonian system
    # system matrices and covariances
    A = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    B = np.array([[0], [1], [0]])
    C = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # measurement noise covariance
    # 1.741=np.std(P_t_hat[1,12:80]-(0.035*(tVec[12:80]-tVec[12])+P_t_hat[1,12]))
    # 0.174=np.std(P_t_hat[2,12:80])
    # R = np.array([[0.288 ** 2, 0, 0], [0, 1.741 ** 2, 0], [0, 0, 0.174 ** 2]])
    R = np.array([[2 ** 2, 0, 0], [0, 5 ** 2, 0], [0, 0, 2 ** 2]])
    # process uncertainty covariance
    Q = np.array([[1 ** 2, 0, 0], [0, 1 ** 2, 0], [0, 0, 1 ** 2]])  # np.matrix(np.zeros((3, 3)))

    # guess of the initial estimate
    x0 = x_hat_init  # np.array([ 511.59460576, -272.16726961, 68.63392779])
    # initial covariance matrix
    P0 = np.asmatrix(np.diag([1, 4, 1]))

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
            X_true[i, k] = x_hat_init[i] + v_hat_init[i] * tVec[k]
            velocity[i, k] = v_hat_init[i]

    # MEASURED(observed) STATES
    # add the measurement noise
    # X_measured = np.zeros((3, np.size(tVec)))
    # for i in range(0, 3):
    #     X_measured[i, :] = X_true[i, :] + np.random.normal(0, np.sqrt(R[i,i]), size=np.size(tVec))
    # TODO: ASSUMPTION: we assume index 12 corresponds to when the object starts to move(initial condition)
    tVec_measured = np.load(log_dir + "/tVec_s1.npy")[12:80]
    P_t_hat = np.load(log_dir + "/P_t_hat_s1.npy")[:, 12:80] + np.array([[0], [0], [33], [0]])
    conf_Z = np.load(log_dir + "/conf_Z_s1.npy")[12:80]

    tVec_measured = ((tVec_measured - tVec_measured[0]) * 1000)  # [ms]
    X_measured = P_t_hat[:3, :]

    # here, we save the noisy outputs such that we can use them in C++ code
    np.savetxt('/home/mahdi/Documents/kalman/myCode/myOutput_x_model_0.csv', X_measured[0, :], delimiter=',')
    np.savetxt('/home/mahdi/Documents/kalman/myCode/myOutput_y_model_0.csv', X_measured[1, :], delimiter=',')
    np.savetxt('/home/mahdi/Documents/kalman/myCode/myOutput_z_model_0.csv', X_measured[2, :], delimiter=',')
    np.savetxt('/home/mahdi/Documents/kalman/myCode/tVec_measured_model_0.csv', tVec_measured, delimiter=',')

    # verify the X_true vector by plotting the results

    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    ax[0].plot(tVec[0:N], X_true[0, 0:N], '-*g', linewidth=1, label='true')
    ax[0].plot(tVec_measured[0:N], X_measured[0, 0:N], '-or', linewidth=1, label='measured')
    ax[0].set_ylabel("x [mm]", fontsize=14)
    ax[0].legend()
    ax[1].plot(tVec[0:N], X_true[1, 0:N], '-*g', linewidth=1, label='true')
    ax[1].plot(tVec_measured[0:N], X_measured[1, 0:N], '-or', linewidth=1, label='measured')
    ax[1].set_ylabel("y [mm]", fontsize=14)
    ax[1].legend()
    ax[2].plot(tVec[0:N], X_true[2, 0:N], '-*g', linewidth=1, label='true')
    ax[2].plot(tVec_measured[0:N], X_measured[2, 0:N], '-or', linewidth=1, label='measured')
    ax[2].set_xlabel("$t_{k}$ [ms]", fontsize=14)
    ax[2].set_ylabel("z [mm]", fontsize=14)
    ax[2].legend()
    plt.tight_layout()
    fig.savefig('data.png', dpi=600)
    plt.show()

    # create a Kalman filter object
    KalmanFilterObject = KalmanFilter(x0, P0, A, B, C, Q, R)
    u = np.array([v_hat_init[1]])
    tVec_measured_rounded = np.round(tVec_measured)
    # simulate online prediction
    for k_measured in range(1, np.size(tVec_measured_rounded)):  # np.arange(np.size(tVec_measured_rounded)):
        print(k_measured)
        # TODO correct for the online application where dt is varying and be know the moment we receive the measurement
        dt = tVec_measured_rounded[k_measured] - tVec_measured_rounded[k_measured - 1]
        KalmanFilterObject.B = np.array([[0], [dt], [0]])
        KalmanFilterObject.propagateDynamics(u)
        KalmanFilterObject.B = np.array([[0], [1], [0]])
        KalmanFilterObject.prediction_aheads(u, dt)
        KalmanFilterObject.computeAposterioriEstimate(X_measured[:, k_measured])

    # extract the state estimates in order to plot the results
    x_hat = []
    y_hat = []
    z_hat = []

    ###############################################################################
    # BEFORE RUNNING THE REMAINING PART OF THE CODE
    # RUN THE C++ CODE THAT IMPLEMENTS THE KALMAN FILTER
    # THE C++ CODE
    # AFTER THAT, YOU CAN RUN THE CODE GIVEN BELOW
    ###############################################################################
    # Here are the C++ estimates
    x_hat_cpp = []
    y_hat_cpp = []
    z_hat_cpp = []

    x_pred_cpp = []
    y_pred_cpp = []
    z_pred_cpp = []
    # Load C++ estimates
    cppEstimates = np.loadtxt("/home/mahdi/Documents/kalman/myCode/myEstimatesAposteriori_model_0.csv", delimiter=",")
    cppPredictions = np.loadtxt("/home/mahdi/Documents/kalman/myCode/X_prediction_ahead_model_0.csv", delimiter=",")
    for j in range(0, np.size(tVec_measured_rounded)):
        # python estimates
        x_hat.append(KalmanFilterObject.estimates_aposteriori[0, j])
        y_hat.append(KalmanFilterObject.estimates_aposteriori[1, j])
        z_hat.append(KalmanFilterObject.estimates_aposteriori[2, j])
        # cpp estimates
        x_hat_cpp.append(cppEstimates[0, j])
        y_hat_cpp.append(cppEstimates[1, j])
        z_hat_cpp.append(cppEstimates[2, j])
    for j in range(0, np.size(tVec)):
        # cpp predictions
        x_pred_cpp.append(cppPredictions[0, j])
        y_pred_cpp.append(cppPredictions[1, j])
        z_pred_cpp.append(cppPredictions[2, j])

    k0 = 0
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    ax[0].plot(tVec, X_true[0, :], '-*g', linewidth=1, markersize=1, label='true')
    ax[0].plot(tVec_measured_rounded[1:], X_measured[0, 1:], '-or', linewidth=1, markersize=5, label='measured')
    ax[0].plot(tVec_measured_rounded, x_hat, 'ob', linewidth=1, markersize=5, label='aposteriori estimated')
    ax[0].plot(tVec_measured_rounded, x_hat_cpp, 'om', linewidth=1, markersize=5, label='aposteriori estimated Cpp')
    ax[0].plot(tVec, x_pred_cpp, '^y', linewidth=1, markersize=5, label='predictions ahead Cpp')
    ax[0].plot(tVec, np.asarray(KalmanFilterObject.X_prediction_ahead[0, :]).squeeze(), '-Dk', linewidth=1,
               markersize=1, label='prediction ahead Python')
    ax[0].set_ylabel("x [mm]", fontsize=14)
    ax[0].legend()
    ax[1].plot(tVec, X_true[1, :], '-*g', linewidth=1, markersize=1, label='true')
    ax[1].plot(tVec_measured_rounded[1:], X_measured[1, 1:], '-or', linewidth=1, markersize=5, label='measured')
    ax[1].plot(tVec_measured_rounded, y_hat, '-ob', linewidth=1, markersize=5, label='aposteriori estimated')
    ax[1].plot(tVec_measured_rounded, y_hat_cpp, '-om', linewidth=1, markersize=5, label='aposteriori estimated Cpp')
    ax[1].plot(tVec, y_pred_cpp, '^y', linewidth=1, markersize=5, label='predictions ahead Cpp')
    ax[1].plot(tVec, np.asarray(KalmanFilterObject.X_prediction_ahead[1, :]).squeeze(), '-Dk', linewidth=1,
               markersize=1, label='prediction ahead')
    ax[1].set_ylabel("y [mm]", fontsize=14)
    ax[1].legend()
    ax[2].plot(tVec, X_true[2, :], '-*g', linewidth=1, markersize=1, label='true')
    ax[2].plot(tVec_measured_rounded[1:], X_measured[2, 1:], '-or', linewidth=1, markersize=5, label='measured')
    ax[2].plot(tVec_measured_rounded, z_hat, '-ob', linewidth=1, markersize=5, label='aposteriori estimated')
    ax[2].plot(tVec_measured_rounded, z_hat_cpp, '-om', linewidth=1, markersize=5, label='aposteriori estimated Cpp')
    ax[2].plot(tVec, z_pred_cpp, '^y', linewidth=1, markersize=5, label='predictions ahead Cpp')
    ax[2].plot(tVec, np.asarray(KalmanFilterObject.X_prediction_ahead[2, :]).squeeze(), '-Dk', linewidth=1,
               markersize=1, label='prediction ahead')
    ax[2].set_xlabel("$t_{k}$ [ms]", fontsize=14)
    ax[2].set_ylabel("z [mm]", fontsize=14)
    ax[2].legend()
    plt.tight_layout()
    fig.savefig('results.png', dpi=600)
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
    np.save(log_dir + "/r_star_model_0.npy", np.asarray(KalmanFilterObject.X_prediction_ahead))
    np.savetxt(log_dir + '/r_star_model_0.csv', np.asarray(KalmanFilterObject.X_prediction_ahead), delimiter=',')
    np.savetxt(log_dir + '/x_star_model_0.csv', np.asarray(KalmanFilterObject.X_prediction_ahead)[0, :], delimiter=',')
    np.savetxt(log_dir + '/y_star_model_0.csv', np.asarray(KalmanFilterObject.X_prediction_ahead)[1, :], delimiter=',')
    np.savetxt(log_dir + '/z_star_model_0.csv', np.asarray(KalmanFilterObject.X_prediction_ahead)[2, :], delimiter=',')
    np.savetxt(log_dir + '/x_star_cpp_model_0.csv', x_pred_cpp, delimiter=',')
    np.savetxt(log_dir + '/y_star_cpp_model_0.csv', y_pred_cpp, delimiter=',')
    np.savetxt(log_dir + '/z_star_cpp_model_0.csv', z_pred_cpp, delimiter=',')
    np.save(log_dir + "/t_model_0.npy", tVec)
    np.savetxt(log_dir + '/t_model_0.csv', tVec, delimiter=',')
    np.save(log_dir + "/tVec_measured_model_0.npy", tVec_measured)
    np.savetxt(log_dir + '/tVec_measured_model_0.csv', tVec_measured, delimiter=',')

    print("end")


def main_model_1(log_dir):
    # initial values for the simulation
    # x_hat_init = np.array([514.5, -269.8, 67.6])  # [mm]
    x_hat_init = np.array([9.5 * 50 + 30 + 5 + 4, -5.5 * 50 + 5, 90 + 3 - 25 + 33, 0, -0.0341,
                           0])  # [mm] manually measured and fixed rigidly+ speed in [mm/1ms]
    # v_hat_init = np.array([0, 3 * 10 / 1000, 0])  # [mm/1ms]
    v_hat_init = np.array([0, 0.0341, 0])  # [mm/1ms]

    # number of discretization time steps
    N = 9066  # dt=1[ms]

    # define the system matrices - Newtonian system
    # system matrices and covariances
    dt = 1
    A = np.matrix(
        [[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    B = np.zeros((6,1))
    C = np.matrix([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

    # measurement noise covariance
    # 1.741=np.std(P_t_hat[1,12:80]-(0.035*(tVec[12:80]-tVec[12])+P_t_hat[1,12]))
    # 0.174=np.std(P_t_hat[2,12:80])
    # R = np.array([[0.288 ** 2, 0, 0], [0, 1.741 ** 2, 0], [0, 0, 0.174 ** 2]])
    R = np.array([[1 ** 2, 0, 0], [0, 2 ** 2, 0], [0, 0, 1 ** 2]])
    # process uncertainty covariance
    Q = np.array(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1e-3 ** 2, 0, 0],
         [0, 0, 0, 0, 10e-3 ** 2, 0],
         [0, 0, 0, 0, 0, 2e-3 ** 2]])  # np.matrix(np.zeros((3, 3)))

    # guess of the initial estimate
    x0 = x_hat_init  # np.array([ 511.59460576, -272.16726961, 68.63392779])
    # initial covariance matrix
    P0 = np.asmatrix(np.diag([1 ** 2, 2 ** 2, 1 ** 2, 1e-3 ** 2, 2e-3 ** 2, 2e-3 ** 2]))

    # time vector for simulation
    tVec = np.linspace(0, (N - 1), N)

    # IDEAL STATES
    # vector used to store the simulated X_true
    X_true = np.zeros((3, np.size(tVec)))
    # velocity = np.zeros((3, np.size(tVec)))
    acceleration = np.zeros((3, np.size(tVec)))
    # simulate the system behavior (discretize system)
    for k in np.arange(np.size(tVec)):
        for i in range(0, 3):
            X_true[i, k] = x_hat_init[i] + v_hat_init[i] * tVec[k]
            # velocity[i, k] = v_hat_init[i]

    # MEASURED(observed) STATES
    # add the measurement noise
    # X_measured = np.zeros((3, np.size(tVec)))
    # for i in range(0, 3):
    #     X_measured[i, :] = X_true[i, :] + np.random.normal(0, np.sqrt(R[i,i]), size=np.size(tVec))
    # TODO: ASSUMPTION: we assume index 12 corresponds to when the object starts to move(initial condition)
    tVec_measured = np.load(log_dir + "/tVec_s1.npy")[12:80]
    P_t_hat = np.load(log_dir + "/P_t_hat_s1.npy")[:, 12:80] + np.array([[0], [0], [33], [0]])
    # conf_Z = np.load(log_dir + "/conf_Z_s1.npy")[12:80]

    tVec_measured = ((tVec_measured - tVec_measured[0]) * 1000)  # [ms]
    X_measured = P_t_hat[:3, :]

    # here, we save the noisy outputs such that we can use them in C++ code
    np.savetxt('myOutput_x_model_1.csv', X_measured[0, :], delimiter=',')
    np.savetxt('myOutput_y_model_1.csv', X_measured[1, :], delimiter=',')
    np.savetxt('myOutput_z_model_1.csv', X_measured[2, :], delimiter=',')

    # verify the X_true vector by plotting the results

    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    ax[0].plot(tVec[0:N], X_true[0, 0:N], '-*g', linewidth=1, label='true')
    ax[0].plot(tVec_measured[0:N], X_measured[0, 0:N], '-or', linewidth=1, label='measured')
    ax[0].set_ylabel("x [mm]", fontsize=14)
    ax[0].legend()
    ax[1].plot(tVec[0:N], X_true[1, 0:N], '-*g', linewidth=1, label='true')
    ax[1].plot(tVec_measured[0:N], X_measured[1, 0:N], '-or', linewidth=1, label='measured')
    ax[1].set_ylabel("y [mm]", fontsize=14)
    ax[1].legend()
    ax[2].plot(tVec[0:N], X_true[2, 0:N], '-*g', linewidth=1, label='true')
    ax[2].plot(tVec_measured[0:N], X_measured[2, 0:N], '-or', linewidth=1, label='measured')
    ax[2].set_xlabel("$t_{k}$ [ms]", fontsize=14)
    ax[2].set_ylabel("z [mm]", fontsize=14)
    ax[2].legend()
    plt.tight_layout()
    fig.savefig('data_model_1.png', dpi=600)
    plt.show()

    ###############################################################################
    # BEFORE RUNNING THE REMAINING PART OF THE CODE
    # RUN THE C++ CODE THAT IMPLEMENTS THE KALMAN FILTER
    # THE C++ CODE
    # AFTER THAT, YOU CAN RUN THE CODE GIVEN BELOW
    ###############################################################################

    # create a Kalman filter object
    KalmanFilterObject = KalmanFilter(x0, P0, A, B, C, Q, R)
    u = np.array([0])
    tVec_measured_rounded = np.round(tVec_measured)
    # simulate online prediction
    for k_measured in range(1, np.size(tVec_measured_rounded)):  # np.arange(np.size(tVec_measured_rounded)):
        print(k_measured)
        # TODO correct for the online application where dt is varying and be know the moment we receive the measurement
        dt = tVec_measured_rounded[k_measured] - tVec_measured_rounded[k_measured - 1]
        KalmanFilterObject.A = np.matrix(
        [[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]])
        KalmanFilterObject.propagateDynamics(u)
        KalmanFilterObject.prediction_aheads(u, dt)
        KalmanFilterObject.computeAposterioriEstimate(X_measured[:, k_measured])

    # extract the state estimates in order to plot the results
    x_hat = []
    y_hat = []
    z_hat = []
    # # Here are the C++ estimates
    # x_hat_cpp = []
    # y_hat_cpp = []
    # z_hat_cpp = []
    # # Load C++ estimates
    # cppEstimates = np.loadtxt("myEstimatesAposteriori.csv", delimiter=",")
    for j in range(0, np.size(tVec_measured_rounded)):  # np.arange(np.size(tVec_measured_rounded)):
        # python estimates
        x_hat.append(KalmanFilterObject.estimates_aposteriori[0, j])
        y_hat.append(KalmanFilterObject.estimates_aposteriori[1, j])
        z_hat.append(KalmanFilterObject.estimates_aposteriori[2, j])
        # # cpp estimates
        # x_hat_cpp.append(cppEstimates[0, j])
        # y_hat_cpp.append(cppEstimates[1, j])
        # z_hat_cpp.append(cppEstimates[2, j])

    k0 = 0
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    ax[0].plot(tVec, X_true[0, :], '-*g', linewidth=1, markersize=1, label='true')
    ax[0].plot(tVec_measured_rounded[1:], X_measured[0, 1:], '-or', linewidth=1, markersize=5, label='measured')
    ax[0].plot(tVec_measured_rounded, x_hat, 'ob', linewidth=1, markersize=5, label='estimated')
    ax[0].plot(tVec, np.asarray(KalmanFilterObject.X_prediction_ahead[0, :]).squeeze(), '-Dk', linewidth=1,
               markersize=1, label='prediction ahead')
    ax[0].set_ylabel("x [mm]", fontsize=14)
    ax[0].legend()
    ax[1].plot(tVec, X_true[1, :], '-*g', linewidth=1, markersize=1, label='true')
    ax[1].plot(tVec_measured_rounded[1:], X_measured[1, 1:], '-or', linewidth=1, markersize=5, label='measured')
    ax[1].plot(tVec_measured_rounded, y_hat, '-ob', linewidth=1, markersize=5, label='estimated')
    ax[1].plot(tVec, np.asarray(KalmanFilterObject.X_prediction_ahead[1, :]).squeeze(), '-Dk', linewidth=1,
               markersize=1, label='prediction ahead')
    ax[1].set_ylabel("y [mm]", fontsize=14)
    ax[1].legend()
    ax[2].plot(tVec, X_true[2, :], '-*g', linewidth=1, markersize=1, label='true')
    ax[2].plot(tVec_measured_rounded[1:], X_measured[2, 1:], '-or', linewidth=1, markersize=5, label='measured')
    ax[2].plot(tVec_measured_rounded, z_hat, '-ob', linewidth=1, markersize=5, label='estimated')
    ax[2].plot(tVec, np.asarray(KalmanFilterObject.X_prediction_ahead[2, :]).squeeze(), '-Dk', linewidth=1,
               markersize=1, label='prediction ahead')
    ax[2].set_xlabel("$t_{k}$ [ms]", fontsize=14)
    ax[2].set_ylabel("z [mm]", fontsize=14)
    ax[2].legend()
    plt.tight_layout()
    fig.savefig('results_model_1.png', dpi=600)
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
    np.save(log_dir + "/r_star_model_1.npy", np.asarray(KalmanFilterObject.X_prediction_ahead))
    np.savetxt(log_dir + '/r_star_model_1.csv', np.asarray(KalmanFilterObject.X_prediction_ahead), delimiter=',')
    np.savetxt(log_dir + '/x_star_model_1.csv', np.asarray(KalmanFilterObject.X_prediction_ahead)[0, :], delimiter=',')
    np.savetxt(log_dir + '/y_star_model_1.csv', np.asarray(KalmanFilterObject.X_prediction_ahead)[1, :], delimiter=',')
    np.savetxt(log_dir + '/z_star_model_1.csv', np.asarray(KalmanFilterObject.X_prediction_ahead)[2, :], delimiter=',')
    np.save(log_dir + "/t_model_1.npy", tVec)
    np.savetxt(log_dir + '/t_model_1.csv', tVec, delimiter=',')

    print("end")


def load_measurements(log_dir):
    tVec = np.load(log_dir + "/tVec_s1.npy")[1:]
    P_c_hat = np.load(log_dir + "/P_c_hat_s1.npy")[:, 1:] * np.array([1e3, 1e3, 1e3, 1]).reshape(4,
                                                                                                 1)  # homogenious in [mm] #[mm]
    conf_Z = np.load(log_dir + "/conf_Z_s1.npy")[1:]
    tVec = ((tVec - tVec[0]) * 1000)  # in [ms]

    plt.figure(figsize=(8, 12))
    plt.subplot(411)
    plt.plot(tVec, P_c_hat[0, :], '-bo')
    plt.ylabel("P_c_hat[0] [mm]")
    plt.subplot(412)
    plt.plot(tVec, P_c_hat[1, :], '-bo')
    plt.ylabel("P_c_hat[1] [mm]")
    plt.subplot(413)
    plt.plot(tVec, P_c_hat[2, :], '-bo')
    plt.ylabel("P_c_hat[2] [mm]")
    plt.subplot(414)
    plt.plot(tVec, conf_Z, '-r*')
    plt.ylabel("conf_Z")
    plt.xlabel("t [ms]")
    plt.savefig(log_dir + "/measurements_s2.png", format="png")
    plt.show()

    r_t2c = np.load("/home/mahdi/Documents/kalman/myCode/logs/measurements/r_t2c_1.npy")
    t_t2c = np.load("/home/mahdi/Documents/kalman/myCode/logs/measurements/t_t2c_1.npy")
    # H_t2c = np.vstack((np.hstack((cv2.Rodrigues(r_t2c[0])[0], t_t2c[0])), np.array([0, 0, 0, 1])))
    t_c2t = -np.matrix(cv2.Rodrigues(r_t2c[0])[0]).T * np.matrix(t_t2c[0])
    R_c2t = np.matrix(cv2.Rodrigues(r_t2c[0])[0]).T
    H_c2t = np.vstack((np.hstack((R_c2t, t_c2t.reshape(3, 1))),
                       np.array([0, 0, 0, 1])))
    P_t_hat = np.zeros_like(P_c_hat)
    for k in range(0, P_c_hat.shape[1]):
        P_t_hat[:, k] = (np.matrix(H_c2t) * np.matrix(P_c_hat[:, k].reshape(4, 1))).squeeze()
    np.save(log_dir + "/P_t_hat_s1.npy", P_t_hat)
    plt.figure(figsize=(8, 16))
    plt.subplot(511)
    plt.plot(tVec, P_t_hat[0, :], '-bo')
    plt.ylabel("P_t_hat[0] [mm]")
    plt.subplot(512)
    plt.plot(tVec, P_t_hat[1, :], '-bo')
    plt.ylabel("P_t_hat[1] [mm]")
    plt.subplot(513)
    plt.plot(tVec, P_t_hat[2, :], '-bo')
    plt.ylabel("P_t_hat[2] [mm]")
    plt.subplot(514)
    plt.plot(tVec, conf_Z, '-r*')
    plt.ylabel("conf_Z")
    plt.subplot(515)
    plt.plot(tVec[1:], np.diff(tVec), '-k^')
    plt.ylabel("dt between measurements [ms]")
    plt.xlabel("t [ms]")
    plt.savefig(log_dir + "/calibrated_measurements_s2.png", format="png")
    plt.show()


if __name__ == "__main__":
    log_dir = "/home/mahdi/Documents/kalman/myCode/logs/measurements"
    main_model_0(log_dir)
    # main_model_1(log_dir)
    # load_measurements(log_dir)
