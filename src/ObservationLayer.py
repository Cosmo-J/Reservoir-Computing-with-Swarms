import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import Ridge
from tqdm import trange
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial import KDTree


class ObservationAndPrediction(ABC):
    READOUT_TYPES=[
        'kernels',  #readout kernels
        'flat',     #flattened state space of the boids
        'COM',      #center of mass of the boids
    ]

    def __init__(self,training,testing):
        self.lorenz = training.get('predator_positions')
        self.testing = testing
        self.training= training

    @abstractmethod
    def get_reservoir_state_vectorised(self,data):
        pass

    def _get_lorenz_targets(self):
        lorenz_x = self.lorenz[:,0]
        targets_x = np.roll(lorenz_x,-1)
        return targets_x[:-1]

    def ridge_prediction(self,state_training,state_testing,ridge_beta=0.1):
        targets_x = self._get_lorenz_targets()

        training_reservoir_state = state_training[:-1]
        ridge = Ridge(alpha=ridge_beta,fit_intercept=False)
        ridge.fit(training_reservoir_state,targets_x)
        prediction = ridge.predict(state_testing[:-1])


        numer = np.mean(targets_x*prediction,axis=0)
        denom = np.sqrt( np.mean(targets_x**2,axis=0) * np.mean(prediction**2,axis=0))
        corr_coef = numer/denom


        return prediction,corr_coef

    def get_plot(self,prediction,corr_coef,x_range,pop_out=False):
        lorenz_x = self.lorenz[:,0]

        fig, ax = plt.subplots(figsize=(20, 6))
        plt.subplots_adjust(bottom=0.2)

        window_size = 2000
        max_index = len(lorenz_x)

        ax.plot(lorenz_x, color='red', label='lorenz_x')
        ax.plot(prediction, color='blue', linestyle='dashed', label='Prediction')
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.xlabel('time')
        plt.ylabel('lorenz_x(t+1)')
        plt.title(f'correlation coeficiant R: {corr_coef}')

        if pop_out:
            ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
            slider = Slider(ax_slider, 'Position', 0, max_index - window_size, valinit=0, valstep=10)

            ax_zoom = plt.axes([0.1, 0.01, 0.8, 0.03])
            zoom_slider = Slider(ax_zoom, 'Window Size', 100, max_index, valinit=window_size, valstep=100)

            def update(val):
                start = int(slider.val)
                end = start + int(zoom_slider.val)
                end = min(end, max_index)
                ax.set_xlim([start, end])
                fig.canvas.draw_idle()

            slider.on_changed(update)
            zoom_slider.on_changed(update)

            ax.set_xlim([0, window_size])
        else:
            ax.set_xlim(x_range)

        return ax



class KernelReadout(ObservationAndPrediction):

    def __init__(self,kernel_number,training,testing):
        super().__init__(training,testing)
        
        self.readout = 'kernels'
        '''number of observation kernels'''
        self.kernel_number = kernel_number
        self.kernels = self.generate_kernels()

    def generate_kernelsOLD(self):
        
        def neighbour_distances(positions, agents:int|list[int] = None):
            '''
                Two functionalities:\n
                    (1) Given no index
                        where N=len(positions), the number of agents
                        if index is left empty, return a matrix of shape [N, N-1]], 
                        where columns are each agent, and the rows are the ordered distance to each neighbour. e.g.
                        [0][33,5,22...,N-1]
                        [1][6,5,13...,N-1]
                        .
                        .
                        .
                        [N][41,17,83...,N-1]

                    (2) If an index (or multiple) are given, a(these) specific row(s) of the matrix described above is(are) returned 
                        i.e. for a(the) specific agent(s)
            '''
            if agents is None:
                indexes = np.array([i for i in range(len(positions))])
            elif isinstance(agents,list):
                indexes = agents
            elif isinstance(agents,int):
                indexes = [agents]
                

            distances = np.array([[None]*len(positions)]*len(positions))
            for i in indexes:
                diff = positions - positions[i]
                distances[i] = np.linalg.norm(diff,axis=1)

            return distances

        def get_closest_neighbours(ordered,closest_num):
            '''
                :params:
                    ordered: 
                        An array of agent distances from a given index x ordered[x] will equal 0 (distance from self).
                        All other elements in the correspond to their distance from the agent represented by index x.
                :return: array length $closest_num$ of agent indexes ordered from closest to furthest
            '''

            #This system is kind of agnostic to duplicate distances. Whilst it's unlikely, in the case of it occuring it wouldn't be a big deal
            dictionaried = {distance:i for i,distance in enumerate(ordered)}
            smallest_to_biggest = np.sort(ordered,kind='quicksort')
                
            closest_agents = [dictionaried[i] for i in smallest_to_biggest[1:closest_num+1]]
            return closest_agents

        kernels = [None] * self.kernel_number
        for m in range(self.kernel_number):
            # below could theoretically be put into one line but seperated it all out for readability
            random_time = np.random.randint(0,self.training['time_steps']) # selects a random t in the range of time steps of the simulation
            xs_at_random_time = self.training['positions'][random_time] # selects the position array at the random time step

            random_agent_index = np.random.randint(0,self.training['boid_count']) # selects a random vector position index at the time step (a given agent)

            neigh_dist_matrix = neighbour_distances(xs_at_random_time,random_agent_index) # create a matrix finding random_agent's distance to all other agents
            fifth_closest = get_closest_neighbours(neigh_dist_matrix[random_agent_index],5)[-1] # find the closest 5 agents to random_agent, index the last one to get the '5th closest' in a sense

            distance_to_fith = np.linalg.norm(xs_at_random_time[random_agent_index] - xs_at_random_time[fifth_closest])
            c_m = xs_at_random_time[random_agent_index] # some random agent
            w_m = distance_to_fith # the fifth closest agent to said random agent
            kernels[m] = [c_m,w_m]
        
        return kernels
    

    def generate_kernels(self):
        '''
            A kernel has a center and a width. 
            The position c_m is found by getting a random position of a boid at a random time.
            The width w_m is the distance between said random boid and it's 5th closest neighber
        '''

        positions = self.training['positions']
        time_steps = self.training['time_steps']
        boid_count = self.training['boid_count']
        
        kernels = [None] * self.kernel_number
        for m in range(self.kernel_number):
            random_time = np.random.randint(0, time_steps) #selects a random t in the range of time steps of the simulation
            xs_at_random_time = positions[random_time] # selects the positions at the random time step

            random_agent_index = np.random.randint(0, boid_count)
            c_m = xs_at_random_time[random_agent_index] # some random agent position
            
            tree = KDTree(xs_at_random_time) 
            distances, indices = tree.query(c_m, k=6)# find the closest 5 agents to random_agent, 6 
            
            #index the last one to get the '5th closest' in a sense
            w_m = distances[5]**2
            kernel_vector = np.append(c_m,w_m) # (x,y,w)
            kernels[m] = kernel_vector
        

        return np.array(kernels)

    def get_reservoir_state_vectorised(self, run, chunk_size=1000):

        kernels = self.kernels
        positions = run['positions']
        velocities = run['velocities']
        T = len(positions)
        
        # where kernels has shape [N,V], N is number of kernels and V is the vector (x_pos,y_pos,width) 
        widths = kernels[:,2] # 3rd item in the vector
        centers = kernels[:,:2] # 1st and 2nd item in the vector


        r1_list, r2_list, r3_list = [], [], []
        
        for chunk_start in trange(0, T, chunk_size, desc="Vectorised"):
            chunk_end = min(chunk_start + chunk_size, T)
            
            pos_chunk = positions[chunk_start:chunk_end]
            vel_chunk = velocities[chunk_start:chunk_end]
            
            diff = pos_chunk[:, :, np.newaxis, :] - centers[np.newaxis, np.newaxis, :, :]
            distances_sq = np.sum(diff**2, axis=-1)
            psi = np.exp(-distances_sq / (2 * widths[np.newaxis, np.newaxis, :]))
            
            r1_list.append(np.sum(psi, axis=1)) # sum of agent positions
            r2_list.append(np.sum(psi * vel_chunk[:, :, 0:1], axis=1)) # sum of x_velocities 
            r3_list.append(np.sum(psi * vel_chunk[:, :, 1:2], axis=1)) # sum of y_velocities
        
        r1 = np.vstack(r1_list)
        r2 = np.vstack(r2_list)
        r3 = np.vstack(r3_list)
        
        return np.concatenate([r1, r2, r3], axis=1)


class FlatReadout(ObservationAndPrediction):
    
    def __init__(self,training,testing):
        super().__init__(training,testing)
        self.readout = 'flat'

    
    def get_reservoir_state_vectorised(self, data):
        x = data['positions']
        v = data['velocities']
        r_i = np.array(np.concatenate([x,v],axis=2))
        r_i = r_i.reshape(r_i.shape[0],r_i.shape[1]*r_i.shape[2])
        
        return r_i