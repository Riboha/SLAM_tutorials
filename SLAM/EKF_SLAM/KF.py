from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
from scipy.stats.distributions import chi2
import os
import matplotlib.pyplot as plt
import imageio
import warnings
warnings.filterwarnings("ignore")


def odom_predict(dt, ekf_state, sigmas):

    P = ekf_state['P']    
    Q = np.diag([sigmas['xy']**2, sigmas['xy']**2, 
                 sigmas['xy']**2, sigmas['xy']**2])
    
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    
    upd_state = A @ ekf_state['x'][0:4]
    upd_covariance = A @ P[0:4,0:4] @ A.T + Q

    ekf_state['x'][0:4] = upd_state
    
    ekf_state['P'][0:4,0:4] = upd_covariance
    ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])
    
    return ekf_state

def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''
    
    ###
    # Implement the GPS update.
    ###

    r = np.array(gps - ekf_state['x'][0:2])
    print(gps)
    P = ekf_state['P']
    R = np.diag([sigmas['gps']**2, sigmas['gps']**2])
    H = np.array([[1, 0],
                  [0, 1]])

    S_inv = np.linalg.inv(P[0:2,0:2] + R)
    
    K = np.matmul(np.matmul(P[0:2,0:2],np.transpose(H)),S_inv)
    ekf_state['x'][:2] = ekf_state['x'][:2] + np.matmul(K,r)
    temp1 = np.identity(ekf_state['x'][:2].size) - np.matmul(K,H)
    ekf_state['P'][0:2,0:2] =  slam_utils.make_symmetric(np.matmul(temp1,P[0:2,0:2]))

    return ekf_state

def laser_measurement_model(ekf_state, landmark_id):
    ''' 
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian. 

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''
    
    ###
    # Implement the measurement model and its Jacobian you derived
    ###
    xv,yv,phi = ekf_state['x'][0], ekf_state['x'][1], ekf_state['x'][2]

    xl,yl = ekf_state['x'][4+2*(landmark_id+1) : 6+2*(landmark_id+1)]

    r_hat = np.sqrt((xl-xv)**2 + (yl-yv)**2)
    theta_hat = np.arctan2((yl-yv),(xl-xv)) - phi
    zhat = (r_hat,slam_utils.clamp_angle(theta_hat))

    H = np.zeros((2,ekf_state['x'].size)) 

    H[0,0] = -(xl - xv)/np.sqrt((xl - xv)**2+(yl - yv)**2)
    H[1,0] = (yl - yv)/((xl - xv)**2+(yl - yv)**2)
    H[0,1] = -(yl - yv)/np.sqrt((xl - xv)**2+(yl - yv)**2)
    H[1,1] = -(xl - xv)/((xl - xv)**2+(yl - yv)**2)
    H[0,2] = 0
    H[1,2] = -1
    H[0,3] = 0
    H[1,3] = 0
    H[0,4] = 0
    H[1,4] = 0
    H[0,5] = 0
    H[1,5] = 0
    H[0,4+2*(landmark_id+1)] = (xl - xv)/np.sqrt((xl - xv)**2+(yl - yv)**2)
    H[1,4+2*(landmark_id+1)] = -(yl - yv)/((xl-xv)**2+(yl-yv)**2)
    H[0,5+2*(landmark_id+1)] = (yl - yv)/np.sqrt((xl - xv)**2+(yl - yv)**2)
    H[1,5+2*(landmark_id+1)] = (xl-xv)/((xl-xv)**2+(yl-yv)**2)

    return zhat, H

def convert_trees(trees):
    converted_trees = []
    # ranges, angles, diameters
    for tree in trees:
        range, angle, _ = tree
        converted_tree = (np.array([np.cos(angle), np.sin(angle)]).T * range)
        converted_trees.append(converted_tree)
    
    return np.array(converted_trees)

def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''

    ###
    # Implement this function.
    ###
    xv,yv,phi = ekf_state['x'][0:3]
    P = ekf_state['P']
    ranges,bearings,dia = tree
    bearings = slam_utils.clamp_angle(bearings)

    xl = xv+ranges*np.cos(bearings+phi)
    yl = yv+ranges*np.sin(bearings+phi)

    upd_state = np.array(np.append(ekf_state['x'], np.array([xl,yl])))
    upd_covariance = np.zeros((upd_state.size,upd_state.size))
    upd_covariance[0:P.shape[0],0:P.shape[1]] = P
    upd_covariance[upd_state.size-1,upd_state.size-1] = 100
    upd_covariance[upd_state.size-2,upd_state.size-2] = 100
    
    ekf_state['num_landmarks'] += 1
    ekf_state['x'] = upd_state
    ekf_state['P'] = slam_utils.make_symmetric(upd_covariance)

    return ekf_state

def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from 
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    ###
    # Implement this function.
    ###
    # print(ekf_state["num_landmarks"])

    P = ekf_state['P']
    R = np.diag([sigmas['range']**2, sigmas['bearing']**2])

    A = np.full((len(measurements),len(measurements)),chi2.ppf(0.96, df=2))                     # 96
    cost_mat = np.full((len(measurements), ekf_state['num_landmarks']), chi2.ppf(0.96, df=2))   # 96

    for k in range(0,len(measurements)):
        for j in range(0,ekf_state['num_landmarks']):
            z_hat,H = laser_measurement_model(ekf_state, j)
            r = np.array(np.array(measurements[k][0:2]) - np.array(z_hat))
            S_inv = np.linalg.inv(np.matmul(np.matmul(H,P),np.transpose(H)) + R)
            MD = np.matmul(np.matmul(np.transpose(r),S_inv), r)
            cost_mat[k,j] = MD

    cost_mat_conc = np.concatenate((cost_mat, A), axis=1)        
    temp1 = np.copy(cost_mat)
    results = slam_utils.solve_cost_matrix_heuristic(temp1)
    assoc = np.zeros(len(measurements),dtype = np.int32)
    costs = []
    for k in range(0, len(results)):
        costs.append(cost_mat_conc[results[k][0],results[k][1]])
        if cost_mat_conc[results[k][0],results[k][1]] > chi2.ppf(0.99, df=2):       # 0.99
            assoc[results[k][0]] = -1
        elif cost_mat_conc[results[k][0],results[k][1]] >= chi2.ppf(0.95, df=2):    # 0.95
            assoc[results[k][0]] = -2
        else:
            assoc[results[k][0]] = results[k][1]
    # plt.hist(costs)
    # plt.show()

    return assoc


def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as 
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''

    ###
    # Implement the EKF update for a set of range, bearing measurements.
    ###

    R = np.diag([sigmas['range']**2, sigmas['bearing']**2])

    for i in range(0,len(trees)):
        if assoc[i]== -2:
            continue
        elif assoc[i]== -1:
            ekf_state = initialize_landmark(ekf_state,trees[i])
            P = ekf_state['P']
            z_hat,H = laser_measurement_model(ekf_state, ekf_state['num_landmarks']-1)
            r = np.array(np.array(trees[i][0:2]) - np.array(z_hat))
            S_inv = np.linalg.inv(np.matmul(np.matmul(H,P),np.transpose(H)) + R) 
            K = np.matmul(np.matmul(P,np.transpose(H)),S_inv)
            ekf_state['x'] = ekf_state['x'] + np.matmul(K,r)
            ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
            temp1 = np.identity(P.shape[0]) - np.matmul(K,H)
            ekf_state['P'] =  slam_utils.make_symmetric(np.matmul(temp1,P))
        else:
            P = ekf_state['P']
            z_hat,H = laser_measurement_model(ekf_state,assoc[i])
            r = np.array(np.array(trees[i][0:2]) - np.array(z_hat))
            S_inv = np.linalg.inv(np.matmul(np.matmul(H,P),np.transpose(H)) + R)
            K = np.matmul(np.matmul(P,np.transpose(H)),S_inv)
            ekf_state['x'] = ekf_state['x'] + np.matmul(K,r)
            ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
            temp2 = np.identity(P.shape[0]) - np.matmul(K,H)
            ekf_state['P'] =  slam_utils.make_symmetric(np.matmul(temp2,P))

    return ekf_state


def run_kf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    plt.gcf().canvas.mpl_connect('key_release_event',
        lambda event: [exit() if event.key == 'escape' else None])
    plt.gcf().gca().set_aspect('equal')
    plt.gcf().tight_layout(pad=0)
    import atexit
    images = []
    atexit.register(lambda: imageio.mimsave(f'./EKF_SLAM_linear_vehicle_gif.gif',
                                            images, fps=40))
    
    last_odom_t = -1
    trees_all = []
    
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    images = []
    for i, event in enumerate(events):
        t = event[1][0]
        
        if i % 1000 == 0:
            print("t = {}".format(t))

        # if event[0] == 'gps':
        #     gps_msmt = event[1][1:]
        #     ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        if event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue

            dt = t - last_odom_t
            ekf_state = odom_predict(dt, ekf_state, sigmas)
            last_odom_t = t

        elif event[0] == 'laser':
            # Laser
            # try:
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            
            
            if len(trees) > 0:
                converted_trees = convert_trees(trees)
                theta = ekf_state['x'][2]
                R = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
                t = ekf_state['x'][0:2]
                trees_all.append((R @ converted_trees.T + t[np.newaxis,:].T).T)

                plt.cla()
                trees_plot = np.vstack(trees_all)
                if not state_history['x'].size < 7:
                    plt.scatter(state_history['x'][-1,0], state_history['x'][-1,1], color='tab:red')
                    for i in range(len(trees)):
                        plt.plot([state_history['x'][-1,0],trees_plot[-(i+1),0]], [state_history['x'][-1,1],trees_plot[-(i+1),1]], color='tab:red', linewidth='0.5')
                    plt.plot(state_history['x'][:,0], state_history['x'][:,1], color='tab:blue', linewidth=1)
                    plt.plot(trees_plot[:,0], trees_plot[:,1], '.g', markersize=0.5)
                    plt.pause(1e-15)
                    
                    # gif
                    plt.gcf().canvas.draw()
                    image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
                    image  = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
                    images.append(image)

            # except:
            #     continue
        
        plt.cla()
        if not state_history['x'].size < 7:
            plt.scatter(state_history['x'][-1,0], state_history['x'][-1,1], color='tab:red')
            plt.plot(state_history['x'][:,0], state_history['x'][:,1], color='tab:blue', linewidth=1)
            plt.pause(1e-15)
            
            # gif
            plt.gcf().canvas.draw()
            image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
            images.append(image)
        
        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:6]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:6,:6])))
        state_history['t'].append(t)
    plt.savefig("EKF_SLAM_linear_vehicle.png")
    return state_history


def main():
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    odo = slam_utils.read_data_file(os.path.join(path, "data/DRS.txt"))
    gps = slam_utils.read_data_file(os.path.join(path, "data/GPS.txt"))
    laser = slam_utils.read_data_file(os.path.join(path, "data/LASER.txt"))

    # collect all events and sort by time
    events = [('odo', x) for x in odo]
    events.extend([('gps', x) for x in gps])
    events.extend([('laser', x) for x in laser])

    events = sorted(events, key = lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": False,
        "plot_raw_laser": True,
        "plot_map_covariances": True

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,              # 0.05
        "phi": 1.0*np.pi/180,     # 0.5
        # Measurement noise
        "gps": 3,
        "range": 0.5,               # 0.5
        "bearing": 5*np.pi/180      # 5*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        # x, y, theta, vx, vy, w
        "x": np.array( [gps[0,1], gps[0,2], 
                    (gps[1,1]-gps[0,1])/(21.968-20.967), (gps[1,2]-gps[0,2])/(21.968-20.967)]),
        "P": np.diag([.1, .1, .1, .1]),
        "num_landmarks": 0
    }

    run_kf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)

if __name__ == '__main__':
    main()
