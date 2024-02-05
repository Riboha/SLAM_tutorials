"""

Particle Filter localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""
import sys
import pathlib
import os
sys.path.append(str(pathlib.Path(__file__).parent))

import math

import matplotlib.pyplot as plt
import numpy as np

from utils.angle import rot_mat_2d

# Estimation parameter of PF
Q = np.diag([0.8, 0.8]) ** 2                # gps
R = np.diag([0.5, np.deg2rad(30.0)]) ** 2   # odom

DT = 0.1  # time tick [s]

# Particle filter parameter
NP = 100  # Number of Particle
NTh = NP / 2.0  # Number of particle for re-sampling

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

    x = F.dot(x) + B.dot(u)

    return x


def gauss_likelihood(x, sigma):
    p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
        math.exp(-x ** 2 / (2 * sigma ** 2))

    return p


def calc_covariance(x_est, px, pw):
    """
    calculate covariance matrix
    see ipynb doc
    """
    cov = np.zeros((3, 3))
    n_particle = px.shape[1]
    for i in range(n_particle):
        dx = (px[:, i:i + 1] - x_est)[0:3]
        cov += pw[0, i] * dx @ dx.T
    cov *= 1.0 / (1.0 - pw @ pw.T)

    return cov


def pf_localization(px, pw, z, u):
    """
    Localization with Particle filter
    """

    for ip in range(NP):
        x = np.array([px[:, ip]]).T
        w = pw[0, ip]

        #  Predict with random input sampling
        ud1 = u[0, 0] + np.random.randn() * R[0, 0] ** 0.5
        ud2 = u[1, 0] + np.random.randn() * R[1, 1] ** 0.5
        ud = np.array([[ud1, ud2]]).T
        x = motion_model(x, ud)

        #  Calc Importance Weight
        for i in range(len(z[:, 0])):
            dx = x[0, 0] - z[i, 1]
            dy = x[1, 0] - z[i, 2]
            pre_z = math.hypot(dx, dy)
            dz = pre_z - z[i, 0]
            w = w * gauss_likelihood(dz, math.sqrt(Q[0, 0]))

        px[:, ip] = x[:, 0]
        pw[0, ip] = w

    pw = pw / pw.sum()  # normalize

    x_est = px.dot(pw.T)
    p_est = calc_covariance(x_est, px, pw)

    N_eff = 1.0 / (pw.dot(pw.T))[0, 0]  # Effective particle number
    if N_eff < NTh:
        px, pw = re_sampling(px, pw)
    return x_est, p_est, px, pw


def re_sampling(px, pw):
    """
    low variance re-sampling
    """

    w_cum = np.cumsum(pw)
    base = np.arange(0.0, 1.0, 1 / NP)
    re_sample_id = base + np.random.uniform(0, 1 / NP)
    indexes = []
    ind = 0
    for ip in range(NP):
        while re_sample_id[ip] > w_cum[ind]:
            ind += 1
        indexes.append(ind)

    px = px[:, indexes]
    pw = np.zeros((1, NP)) + 1.0 / NP  # init weight

    return px, pw


def plot_covariance_ellipse(x_est, p_est):  # pragma: no cover
    p_xy = p_est[0:2, 0:2]
    eig_val, eig_vec = np.linalg.eig(p_xy)

    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)

    # eig_val[big_ind] or eiq_val[small_ind] were occasionally negative
    # numbers extremely close to 0 (~10^-20), catch these cases and set the
    # respective variable to 0
    try:
        a = math.sqrt(eig_val[big_ind])
    except ValueError:
        a = 0

    try:
        b = math.sqrt(eig_val[small_ind])
    except ValueError:
        b = 0

    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
    fx = rot_mat_2d(angle) @ np.array([[x, y]])
    px = np.array(fx[:, 0] + x_est[0, 0]).flatten()
    py = np.array(fx[:, 1] + x_est[1, 0]).flatten()
    plt.plot(px, py, "--r")

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
    gt_traj, odom, gps = read_data(os.path.join(str(pathlib.Path(__file__).parent), "data"))

    # State Vector [x y yaw v]'
    x_est = np.array([[gt_traj[0][0,0]],
                    [gt_traj[0][1,0]],
                    [gt_traj[0][2,0]],
                    [odom[0][0,0]]])
    x_true = gt_traj[0]

    px = np.repeat(x_est, NP, axis=-1)  # particles
    pw = np.zeros((1, NP)) + 1.0 / NP   # Particle weight

    # history
    h_x_est = x_est
    h_x_true = x_true

    for i in range(1, len(gt_traj)-1):
        u = odom[i]
        z = gps[i]
        x_true = gt_traj[i]
        
        for ip in range(NP):
            x = np.array([px[:, ip]]).T
            w = pw[0, ip]
        
            #  Predict with random input sampling
            ud1 = u[0, 0] #+ np.random.randn() * R[0, 0] ** 0.5
            ud2 = u[1, 0] #+ np.random.randn() * R[1, 1] ** 0.5
            ud = np.array([[ud1, ud2]]).T
            RR = np.diag([0.1, 0.1, np.deg2rad(1.0), 0.5])
            x = motion_model(x, ud) + RR @ np.random.randn(4,1)
            
            # Calc Importance Weight
            for ii in range(len(z[:, 0])):
                dx = x[0, 0] - z[0, 0]
                dy = x[1, 0] - z[1, 0]
                dz = np.sqrt(dx**2 + dy**2)
                w = w * gauss_likelihood(dz, math.sqrt(Q[0, 0]))

            px[:, ip] = x[:, 0]
            pw[0, ip] = w

        pw = pw / pw.sum()  # normalize
        x_est = px.dot(pw.T)
        p_est = calc_covariance(x_est, px, pw)
        
        N_eff = 1.0 / (pw.dot(pw.T))[0, 0]  # Effective particle number
        if N_eff < NTh:
            px, pw = re_sampling(px, pw)
        
        # store data history
        h_x_est = np.hstack((h_x_est, x_est))
        h_x_true = np.hstack((h_x_true, x_true))

        if show_animation:
            plt.cla()
            plt.plot(px[0, :], px[1, :], ".r")
            plt.plot(np.array(h_x_true[0, :]).flatten(),
                     np.array(h_x_true[1, :]).flatten(), "-b")
            plt.plot(np.array(h_x_est[0, :]).flatten(),
                     np.array(h_x_est[1, :]).flatten(), "-r")
            plot_covariance_ellipse(x_est, p_est)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(1e-15)
    ate_rmse = cal_ate_rmse(h_x_true, h_x_est)
    print(f"ATE RMSE[cm] : {ate_rmse:.3f}")
    plt.title(f"Particle Filter Localization\nATE RMSE[cm] : {ate_rmse:.3f}")
    plt.pause(1e-15)
    plt.savefig("/home/lair99/slam_example/PF.png")
    plt.show()

if __name__ == '__main__':
    main()
