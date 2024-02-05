import sys
import pathlib
import os

sys.path.append(str(pathlib.Path(__file__).parent))

import math
import matplotlib.pyplot as plt
import numpy as np

from utils.plot import plot_covariance_ellipse

# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    0.5  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([0.8, 0.8]) ** 2  # Observation x,y position covariance

DT = 0.1  # time tick [s]

show_animation = True


def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


def jacob_f(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH


def ekf_estimation(xEst, P_t, z, u):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ P_t @ jF.T + Q

    #  Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    P_t = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, P_t


def read_data(data_path):
    gt_traj_txt = open(os.path.join(data_path, "gt_traj.txt"), "r")
    u_txt = open(os.path.join(data_path, "odom.txt"), "r")
    gps_txt = open(os.path.join(data_path, "gps.txt"), "r")
    
    gt_traj_data = gt_traj_txt.readlines()
    u_data = u_txt.readlines()
    gps_data = gps_txt.readlines()
    gt_traj = []
    u = []
    gps = []
    
    for i in range(len(gt_traj_data)):
        gt_traj_data_ = gt_traj_data[i].split()
        data = np.array([[float(gt_traj_data_[0])],
                         [float(gt_traj_data_[1])],
                         [float(gt_traj_data_[2])]])
        gt_traj.append(data)
        
        u_data_ = u_data[i].split()
        data = np.array([[float(u_data_[0])],
                         [float(u_data_[1])]])
        u.append(data)
    
        gps_data_ = gps_data[i].split()
        data = np.array([[float(gps_data_[0])],
                         [float(gps_data_[1])]])
        gps.append(data)

    return gt_traj, u, gps

def cal_ate_rmse(gt_traj, est_traj):
    gt_traj = np.array(gt_traj[:2,:])
    est_traj = np.array(est_traj[:2,:])
    
    ate_rmse = np.mean((gt_traj - est_traj)**2) * 100.0
    return ate_rmse

def main():
    # print(__file__ + " start!!")
    gt_traj, odom, gps = read_data(os.path.join(str(pathlib.Path(__file__).parent), "data"))
    
    # State Vector x : [x y yaw v]'
    xEst = np.array([[gt_traj[0][0,0]],
                    [gt_traj[0][1,0]],
                    [gt_traj[0][2,0]],
                    [odom[0][0,0]]])
    PEst = np.eye(4)

    # history
    x_history = xEst
    z_history = gps[0]
    x_True = gt_traj[0]
    
    for i in range(1, len(gt_traj)-1):
        # x_{t-1}
        u = odom[i]
        z = gps[i]
        
        # Prediction step
        xPred = motion_model(xEst, u)
        jF = jacob_f(xEst, u)
        PPred = jF @ PEst @ jF.T + Q
        
        # Correction step
        jH = jacob_h()
        zPred = observation_model(xPred)
        y = z - zPred
        S = jH @ PPred @ jH.T + R
        K = PPred @ jH.T @ np.linalg.inv(S)
        xEst = xPred + K @ y
        PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
        
        # store data history
        x_history = np.hstack((x_history, xEst))
        z_history = np.hstack((z_history, z))
        x_True = np.hstack((x_True, gt_traj[i]))
        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            # plt.plot(z_history[0, :], z_history[1, :], ".g")
            plt.plot(x_True[0, :],
                     x_True[1, :], "-b")
            plt.plot(x_history[0, :],
                     x_history[1, :], "-r")
            plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(1e-15)
    ate_rmse = cal_ate_rmse(x_True, x_history)
    print(f"ATE RMSE[cm] : {ate_rmse:.3f}")
    plt.title(f"EKF Localization\nATE RMSE[cm] : {ate_rmse:.3f}")
    plt.pause(1e-15)
    plt.savefig("/home/lair99/slam_example/EKF.png")
    plt.show()

if __name__ == '__main__':
    main()
