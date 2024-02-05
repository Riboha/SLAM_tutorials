#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import icp
import g2o
import pose_graph
import scipy
import imageio
import os
import slam_utils
import copy
import warnings
warnings.filterwarnings("ignore")

def hessian_matrix(hessian_fun):
    hessian = np.ndarray((3, 3))
    for i in range(3):
        for j in range(3):
            hessian[i, j] = hessian_fun(i, j)
    return hessian

def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

def odom_update(u, dt, prev_init_pose, vehicle_params):
    Ve = u[0]
    alpha = slam_utils.clamp_angle(u[1])

    H = vehicle_params['H']
    L = vehicle_params['L']
    a = vehicle_params['a']
    b = vehicle_params['b']

    Vc = Ve/(1-np.tan(alpha)*H/L)
    
    # phi = np.arctan2(prev_init_pose[1,0], prev_init_pose[0,0])
    
    d_x = dt*(Vc- Vc/L*np.tan(alpha)*b)
    d_y = dt*(Vc/L*np.tan(alpha)*a)
    d_theta = dt*Vc/L*np.tan(alpha)
    
    d_pose = np.array([[np.cos(d_theta), -np.sin(d_theta), d_x],
                       [np.sin(d_theta),  np.cos(d_theta), d_y],
                       [0., 0., 1.]])
    
    updated_init_pose = d_pose @ prev_init_pose
    
    return updated_init_pose


def convert_scan(scan):
    angles = np.array(range(361))*np.pi/360 - np.pi/2
    converted_scans = (np.array([np.cos(angles), np.sin(angles)]).T * scan[:, np.newaxis])
    filter = np.bitwise_and(scan<40.0, scan>3.0)
    converted_scans = converted_scans[filter]
    # plt.scatter(converted_scans[:,0], converted_scans[:,1])
    # plt.show()
    return converted_scans

def main():
    save_gif = True
    if save_gif:
        plt.gcf().canvas.mpl_connect('key_release_event',
            lambda event: [exit() if event.key == 'escape' else None])
        plt.gcf().gca().set_aspect('equal')
        plt.gcf().tight_layout(pad=0)
        import atexit
        images = []
        atexit.register(lambda: imageio.mimsave(f'./graph_SLAM_gif.gif',
                                                images, fps=10))
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    odoms = slam_utils.read_data_file(os.path.join(path, "data/DRS.txt"))
    gps = slam_utils.read_data_file(os.path.join(path, "data/GPS.txt"))
    lasers = slam_utils.read_data_file(os.path.join(path, "data/LASER.txt"))
    
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in lasers])
    events.extend([('odom', x) for x in odoms])

    events = sorted(events, key = lambda event: event[1][0])

    # Starting point
    optimizer = pose_graph.PoseGraphOptimization()
    pose = np.array([[np.cos(36*np.pi/180), -np.sin(36*np.pi/180), gps[0,1]],
                              [np.sin(36*np.pi/180), np.cos(36*np.pi/180), gps[0,2]],
                              [0, 0, 1]])
    optimizer.add_vertex(0, g2o.SE2(g2o.Isometry2d(pose)), True)

    vertex_idx = 1
    registered_points = []

    max_x = -float('inf')
    max_y = -float('inf')
    min_x = float('inf')
    min_y = float('inf')

    robot_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    last_odom_t = 0
    prev_points = []
    init_pose = np.eye(3,3)
    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))

        # Odometry input
        if event[0] == 'odom':
            # Initialize
            if last_odom_t == 0:
                last_odom_t = t
            else:
                u = event[1][1:]
                dt = t - last_odom_t
                if dt == 0:
                    continue
                else:
                    init_pose = odom_update(u, dt, init_pose, robot_params)
                    last_odom_t = t

        # LiDAR input, scan matching
        if event[0] == 'laser' and (np.linalg.norm(init_pose[:2,2]) > 2.0 or abs(np.arctan2(init_pose[1,0], init_pose[0,0])) > 0.2):   #0.2
            if last_odom_t == 0:
                continue
            if len(prev_points)==0:
                scan = event[1][1:]
                prev_points = convert_scan(scan)
                registered_points.append(prev_points)
                continue
            
            # point to point ICP
            scan = event[1][1:]
            current_points = convert_scan(scan)
            with np.errstate(all='raise'):
                try:
                    tran, distances, iter, cov = icp.icp(
                        current_points, prev_points, init_pose,
                        max_iterations=80, tolerance=1e-5,
                        max_corres_distance = 0.5)
                except Exception as e:
                    continue
            
            # Add new node/edge
            pose = np.matmul(pose, tran)
            optimizer.add_vertex(vertex_idx, g2o.SE2(g2o.Isometry2d(pose)))
            rk = g2o.RobustKernelDCS()
            information = np.linalg.inv(cov)
            optimizer.add_edge([vertex_idx-1, vertex_idx],
                                g2o.SE2(g2o.Isometry2d(tran)),
                                information, robust_kernel=rk)

            # For visualization/next step
            registered_points.append(current_points)
            last_odom_t = t
            prev_points = copy.deepcopy(current_points)
            init_pose = np.eye(3,3)
            
            # Loop Closure
            if vertex_idx > 10 and not vertex_idx % 10:
                new_graph = False
                poses = [optimizer.get_pose(idx).to_vector()[0:2]
                        for idx in range(vertex_idx-1)]
                
                kd = scipy.spatial.cKDTree(poses)
                x, y, _ = optimizer.get_pose(vertex_idx).to_vector()
                idxs = kd.query_ball_point(np.array([x, y]), r=10.)
                for idx in idxs:
                    A = registered_points[idx]
                    with np.errstate(all='ignore'):
                        try:
                            tran, distances, iter, cov = icp.icp(
                                A, current_points, np.eye(3,3),
                                max_iterations=80, tolerance=1e-5,
                                max_corres_distance = 0.5)
                        except Exception as e:
                            continue
                    
                    if np.mean(distances) < 1.1:
                        new_graph = True
                        rk = g2o.RobustKernelDCS()
                        optimizer.add_edge([vertex_idx, idx],
                                            g2o.SE2(g2o.Isometry2d(tran)),
                                            np.eye(3), robust_kernel=rk)
                if new_graph:
                    optimizer.optimize()
                    pose = optimizer.get_pose(vertex_idx).to_isometry().matrix()

            # Visualize
            traj = []
            point_cloud = []
            draw_last = float('inf')

            for idx in range(max(0, vertex_idx-draw_last), vertex_idx):
                x = optimizer.get_pose(idx)
                r = x.to_isometry().R
                t = x.to_isometry().t
                filtered = registered_points[idx]
                filtered = filtered[np.linalg.norm(filtered, axis=1) < 80]
                point_cloud.append((r @ filtered.T + t[:, np.newaxis]).T)
                traj.append(x.to_vector()[0:2])
            
            point_cloud = np.vstack(point_cloud)
            xyreso = 0.01 # Map resolution (m)
            point_cloud = (point_cloud / xyreso).astype('int')
            point_cloud = np.unique(point_cloud, axis=0)
            point_cloud = point_cloud * xyreso

            current_max = np.max(point_cloud, axis=0)
            current_min = np.min(point_cloud, axis=0)
            max_x = max(max_x, current_max[0])
            max_y = max(max_y, current_max[1])
            min_x = min(min_x, current_min[0])
            min_y = min(min_y, current_min[1])
            traj = np.array(traj)
            
            plt.cla()
            plt.axis([min_x, max_x, min_y, max_y])
            plt.scatter(traj[-1,0], traj[-1,1], color='tab:red')
            plt.plot(traj[:, 0], traj[:, 1], color='tab:blue', linewidth=1)
            plt.plot(point_cloud[:, 0], point_cloud[:, 1], '.g', markersize=0.1)
            plt.pause(1e-15)

            if save_gif:
                plt.gcf().canvas.draw()
                image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
                image  = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
                images.append(image)

            vertex_idx += 1
    plt.savefig(f'graph_SLAM_result.png')
if __name__=="__main__":
    main()