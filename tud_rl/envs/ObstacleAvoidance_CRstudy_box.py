import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
import matplotlib.patches as patches




class ObstacleAvoidance_CRstudy_box(gym.Env):
    """Class environment with initializer, step, reset and render method."""

   
    
    def __init__(self, POMDP_type = "MDP", N_obst = 1, box_size = 25, mode="train"):
        
        # ----------------------------- settings and hyperparameter -----------------------------------------
        assert POMDP_type in ["MDP", "RV", "FL"], "Unknown MDP/POMDP specification."
        self.POMDP_type  = POMDP_type 
        self.mode = mode
        self.N_obst = N_obst

        # constants
        self.box_size = box_size
        self.m = 15
        self.J = 30
        self.deltaT = 0.2
        self.Radius = 3

        # dynamic constraints
        self.u_max = 2
        self.r_max = 0.2

        self.u_norm = 1.5
        self.r_norm = 0.5
        self.d_norm = 15

        
        # rendering
        self.plot_delay = 0.001

        
        # --------------------------------  gym inherits ---------------------------------------------------
        if self.POMDP_type == "RV":
            num_vessel_obs = 2
        else:
            num_vessel_obs = 4

        super(ObstacleAvoidance_CRstudy_box, self).__init__()
        self.observation_space = spaces.Box(low=np.full(( 1, (self.N_obst * num_vessel_obs + 2)), -1, dtype=np.float32)[0],
                                            high=np.full((1, (self.N_obst * num_vessel_obs + 2)), 1,  dtype=np.float32)[0])
        self.action_space = spaces.Box(low=np.array([-1, -1], dtype=np.float32), 
                                       high=np.array([1, 1], dtype=np.float32))
        
        # --------------------------------- custom inits ---------------------------------------------------

    def reset(self):
        """Resets environment to initial state."""
        self.current_timestep = 0
        self.reward = 0
        self.action = [0, 0]
        self._set_dynamics()
        self._set_state()

        return self.state
   
    def _set_dynamics(self):
        """Initializes positions, velocity and acceleration of agent and vessels."""
        # agent initialization             
        self.x = 0
        self.y = 0
        self.phi = 0

        self.u = self.u_max/2
        self.v = 0
        self.r = 0

        # obstacle initialization
        self.x_obst  = np.random.uniform(-self.box_size, self.box_size, self.N_obst) 
        self.y_obst  = np.random.uniform(-self.box_size, self.box_size, self.N_obst) 
        self.vx_obst = np.random.uniform(-self.u_max,self.u_max, self.N_obst) 
        self.vy_obst = np.random.uniform(-self.u_max,self.u_max, self.N_obst) 

        self.phi_obst = np.arctan2(self.vy_obst,self.vx_obst)
        self.u_obst = np.sqrt(np.power(self.vx_obst,2) + np.power(self.vy_obst,2))

        self.d_obst = np.sqrt(np.power(self.x - self.x_obst,2) + np.power(self.y - self.y_obst,2))


    def _set_state(self):
        """Sets state"""    

        # compute angles
        theta = np.arctan2(self.y_obst-self.y, self.x_obst-self.x) - self.phi       # direction of obstacle position in body frame
        #theta2 = self.phi_obst - self.phi                                          # moving direction of obstacle with respect to moving agent
        theta2 = np.arctan2(self.y-self.y_obst, self.x-self.x_obst) - self.phi_obst # moving direction of agent with respect to obstacle

        # compute sorting index array for asceding euclidean distance
        idx = np.argsort(self.d_obst)

        self.state = np.array([self.u/self.u_norm,
                               self.r/self.r_norm])

        self.state = np.append(self.state,  self.d_obst[idx]/self.d_norm)
        self.state = np.append(self.state,  theta[idx]/np.pi)

        # POMDP specs
        if self.POMDP_type == "MDP":
            v_obs = np.array([(self.u_obst[idx])/self.u_norm,
                               theta2[idx]/np.pi])
            self.state = np.append(self.state, v_obs)
        
      
    def step(self, action):
        """Takes an action and performs one step in the environment.
        Returns reward, new_state, done."""

        #action = [0.8,0.8]
       
        self._move_obst()
        self._move_agent(action)

        self.action = action
        self._set_state()
        done = self._reward()


        self.current_timestep += 1
        
        return self.state, self.reward, done, {}
    
    def _move_obst(self):
        """Updates positions and velocities of obstacles"""
        
        self.x_obst = self.x_obst + self.vx_obst * self.deltaT
        self.y_obst = self.y_obst + self.vy_obst * self.deltaT

        # mirroring at box borders
        for i in range(self.N_obst):
            if self.x_obst[i] < self.x - self.box_size:
                #self.vx_obst[i] = np.abs(self.vx_obst[i])
                #self.x_obst[i] = self.x - self.box_size
                self.x_obst[i] = self.x + self.box_size
            elif self.x_obst[i] > self.x + self.box_size:
                #self.vx_obst[i] = - np.abs(self.vx_obst[i])
                #self.x_obst[i] = self.x + self.box_size
                self.x_obst[i] = self.x - self.box_size

            if self.y_obst[i] < self.y - self.box_size:
                #self.vy_obst[i] = np.abs(self.vy_obst[i])
                #self.y_obst[i] = self.y - self.box_size
                self.y_obst[i] = self.y + self.box_size
            elif self.y_obst[i] > self.y + self.box_size:
                #self.vy_obst[i] = - np.abs(self.vy_obst[i])     
                #self.y_obst[i] = self.y + self.box_size
                self.y_obst[i] = self.y - self.box_size

    
    def _move_agent(self, action):
        """Update agent"""    

        self.r = self.r + 1/self.J * action[0] * self.deltaT
        self.u = self.u + 1/self.m * action[1] * self.deltaT

        self.r = np.clip(self.r, -self.r_max, self.r_max)
        self.u = np.clip(self.u, 0, self.u_max)

        x_dot = np.cos(self.phi) * self.u
        y_dot = np.sin(self.phi) * self.u

        self.x = self.x + x_dot * self.deltaT
        self.y = self.y + y_dot * self.deltaT
        self.phi = self.phi + self.r * self.deltaT

        # distance to obstacles
        self.d_obst_old = self.d_obst
        self.d_obst = np.sqrt(np.power(self.x - self.x_obst,2) + np.power(self.y - self.y_obst,2))
    
    def _reward(self):
        if any(self.d_obst < self.Radius):
            self.reward = -10
            done = False
        else:
            self.reward = 0.1
            done = False
        return done
        

    
    def render(self, agent_name=None):
        """Renders the current environment."""

        # plot every nth timestep
        if self.current_timestep % 2 == 0: 

            # check whether figure has been initialized
            if len(plt.get_fignums()) == 0:
                self.fig = plt.figure(figsize=(17, 10))
                self.gs  = self.fig.add_gridspec(2, 2)
                self.ax0 = self.fig.add_subplot(self.gs[0, 0]) # ship
                self.ax01 = self.fig.add_subplot(self.gs[0, 0]) # ship
                self.ax1 = self.fig.add_subplot(self.gs[1, 0]) # state
                self.ax2 = self.fig.add_subplot(self.gs[1, 1]) # reward
                self.ax3 = self.fig.add_subplot(self.gs[0, 1]) # action
                self.ax2.old_time = 0
                self.ax2.old_reward = 0
                self.ax3.old_time = 0
                self.ax3.old_action = 0
                plt.ion()
                plt.show()
            
            # ---- ACTUAL SHIP MOVEMENT ----
            # clear prior axes, set limits and add labels and title
            
            self.ax0.clear()
            self.ax0.set_xlim(self.x - self.box_size, self.x + self.box_size)
            self.ax0.set_ylim(self.y - self.box_size, self.y + self.box_size)
            self.ax0.set_xlabel("x")
            self.ax0.set_ylabel("y")
            if agent_name is not None:
                self.ax0.set_title(agent_name)

            # set agent and vessels
            self.ax0.scatter(self.x, self.y, s = self.Radius/2, color = "red")
            self.ax0.scatter(self.x_obst, self.y_obst, s = 2, color = "green")   
            circ = patches.Circle((self.x, self.y), radius=self.Radius/2, edgecolor='blue', facecolor='none', alpha=0.3)
            self.ax0.add_patch(circ)
            for i in range(self.N_obst):
                circ2 = patches.Circle((self.x_obst[i], self.y_obst[i]), radius=self.Radius/2, edgecolor='blue', facecolor='none', alpha=0.3)
                self.ax0.add_patch(circ2)

            # visualize path
            if self.current_timestep == 0:
                self.ax01.clear()
                self.ax01.old_x = 0
                self.ax01.old_y = 0
                self.ax01.patch.set_alpha(0.5)
            self.ax01.set_xlim(self.x - self.box_size, self.x + self.box_size)
            self.ax01.set_ylim(self.y - self.box_size, self.y + self.box_size)                
            self.ax01.plot([self.ax01.old_x, self.x], [self.ax01.old_y, self.y], color = "black")
            self.ax01.old_x = self.x
            self.ax01.old_y = self.y

            
            # ---- STATE PLOT ----
            # clear prior axes, set limits
            if self.current_timestep == 0:
                self.ax1.clear()
                self.ax1.old_time = 0
                self.ax1.old_state = self.state     
                self.ax1.set_xlim(0, 200)
                self.ax1.set_ylim(-2,2)
                self.ax1.legend(loc="upper right")
                self.ax1.labels = np.array(['u', 'r'])
                self.ax1.labels = np.append(self.ax1.labels, np.repeat('d_obst',self.N_obst))
                self.ax1.labels = np.append(self.ax1.labels, np.repeat('theta',self.N_obst))
                self.ax1.labels = np.append(self.ax1.labels, np.repeat('u_rel',self.N_obst))
                self.ax1.labels = np.append(self.ax1.labels, np.repeat('theta2',self.N_obst))

                self.ax1.colors = np.array(['lightcoral', 'red'])
                self.ax1.colors = np.append(self.ax1.colors, np.repeat('green',self.N_obst))
                self.ax1.colors = np.append(self.ax1.colors, np.repeat('blue',self.N_obst))
                self.ax1.colors = np.append(self.ax1.colors, np.repeat('plum',self.N_obst))
                self.ax1.colors = np.append(self.ax1.colors, np.repeat('peru',self.N_obst))
                for old_statevar, statevar, c, l in zip(self.ax1.old_state, self.state,  self.ax1.colors, self.ax1.labels):
                    self.ax1.plot([self.ax1.old_time, self.current_timestep], [old_statevar, statevar], color= c, label=l)
           
            #for old_statevar, statevar, c in zip(self.ax1.old_state, self.state,  self.ax1.colors):
            #    self.ax1.plot([self.ax1.old_time, self.current_timestep], [old_statevar, statevar], color= c)
            self.ax1.old_time = self.current_timestep
            self.ax1.old_state = self.state            
            
            self.ax1.set_xlabel("timestep")
            self.ax1.set_ylabel("state var")
            self.ax1.legend(loc="upper right")

 

            # ---- REWARD PLOT ----
            if self.current_timestep == 0:
                self.ax2.clear()
                self.ax2.old_time = 0
                self.ax2.old_reward = 0
            self.ax2.set_xlim(1, 500)
            self.ax2.set_ylim(-1,0)
            self.ax2.set_xlabel("Timestep in episode")
            self.ax2.set_ylabel("Reward")
            self.ax2.plot([self.ax2.old_time, self.current_timestep], [self.ax2.old_reward, self.reward], color = "black")
            self.ax2.old_time = self.current_timestep
            self.ax2.old_reward = self.reward

            # ---- ACTION PLOT ----
            if self.current_timestep == 0:
                self.ax3.clear()
                self.ax3.old_time = 0
                self.ax3.old_action = np.array([0, 0])
            self.ax3.set_xlim(1, 500)
            self.ax3.set_ylim(-1,1)
            self.ax3.set_xlabel("Timestep in episode")
            self.ax3.set_ylabel("Agent a_y")            
            self.ax3.plot([self.ax3.old_time, self.current_timestep], [self.ax3.old_action[0], self.action[0]], label="tau", color="blue")
            self.ax3.plot([self.ax3.old_time, self.current_timestep], [self.ax3.old_action[1], self.action[1]], label="f", color="red")

            if self.current_timestep == 0:
                self.ax3.legend()
            self.ax3.old_time = self.current_timestep
            self.ax3.old_action = self.action
            
            # delay plotting for ease of user
            plt.pause(self.plot_delay)
