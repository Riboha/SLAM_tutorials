import numpy as np
import pygame
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pygame.init()
    background = pygame.display.set_mode((480, 360))
    pygame.display.set_caption('Controller')
    running = True

    # gt
    state = np.zeros((3,1))     # x,y,theta
    u = np.zeros((2,1))         # v,w
    # noisy data
    u_noisy = np.zeros((2,1))
    gps = np.zeros((2,1))
    dt = 0.1
    INPUT_NOISE = np.diag([0.5, np.deg2rad(30.0)]) ** 2
    GPS_NOISE = np.diag([0.8, 0.8]) ** 2
    
    # initial value
    u[0,-1] = 5.
    u[1,-1] = 0.2
    
    # file
    gt_traj_txt = open("Localization/data/gt_traj.txt", "w")
    noisy_u_file = open("Localization/data/odom.txt", "w")
    noisy_gps_file = open("Localization/data/gps.txt", "w")
    
    for i in range(300):
        plt.cla()
        plt.plot(state[0,:], state[1,:])
        plt.pause(1e-15)
    
    while running:
        # u_t
        u_t = np.zeros((2,1))
        u_t[:,0] = u[:,-1]
        
        # state_{t-1}
        state_t1 = np.zeros((3,1))
        state_t1[:,0] = state[:,-1]
        
        # get control info
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running=False
                if event.key == pygame.K_UP:
                    u_t[0,0] += 0.5
                elif event.key == pygame.K_DOWN:
                    u_t[0,0] -= 0.5
                elif event.key == pygame.K_RIGHT:
                    u_t[1,0] -= 0.2
                elif event.key == pygame.K_LEFT:
                    u_t[1,0] += 0.2
        u_t[0,0] += 0.1 * np.random.randn(1)
        print(u_t)
        
        # update
        B = np.array([[dt * np.cos(state_t1[2,0]), 0],
                      [dt * np.sin(state_t1[2,0]), 0],
                      [0, dt]])
        
        state_t = np.zeros((3,1))
        state_t = state_t1 + B@u_t
        
        # gt
        state = np.hstack((state, state_t))
        u = np.hstack((u, u_t))
        
        # noisy data
        # control input
        u_t_noisy = u_t + INPUT_NOISE@np.random.randn(2, 1)
        u_noisy = np.hstack((u_noisy, u_t_noisy))
        
        # gps
        state_t_noisy = state_t[:2,:] + GPS_NOISE@np.random.randn(2, 1)
        gps = np.hstack((gps, state_t_noisy))
        
        # write
        gt_traj_txt.write(f"{state[0,-1]} {state[1,-1]} {state[2,-1]}\n")
        noisy_u_file.write(f"{u_noisy[0,-1]} {u_noisy[1,-1]}\n")
        noisy_gps_file.write(f"{gps[0,-1]} {gps[1,-1]}\n")
        
        plt.cla()
        plt.plot(state[0,:], state[1,:], 'b')
        # plt.plot(gps[0,:], gps[1,:], 'r')
        plt.pause(dt)
        
    gt_traj_txt.close()
    noisy_u_file.close()
    noisy_gps_file.close()
    pygame.quit()