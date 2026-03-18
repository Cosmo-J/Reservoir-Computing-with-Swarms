import numpy as np
from sklearn.linear_model import Ridge
from tqdm import trange
from src.SimSaverLoader import SimSaverLoader as ssl
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

class PropertyAnalysis:
    '''
    Used to analysise the properties of a reservoir
    '''

    def __init__(self,simulations):
        self.simulations = simulations

        self.state_vectors = [] #per replica
        self.covariance_matricies = []# per replica
        self.eigen_vectors = []# per replica
        self.eigen_values = []# per replica
        self.normalisation_transforms = []# per replica
        self.state_vectors_normalised = []

        for s in simulations:
            s_state_vector = self.__calc_state_vector(s)
            self.state_vectors.append(s_state_vector)

            s_covariance_matrix = self.__calc_covariance_matrix(s_state_vector)
            self.covariance_matricies.append(s_covariance_matrix)

            s_eigen_values, s_eigen_vectors =  np.linalg.eigh(s_covariance_matrix)
            sigma_squared = np.diagflat(s_eigen_values)

            self.eigen_vectors.append(s_eigen_vectors)
            self.eigen_values.append(sigma_squared)

            s_norm_trans = self.__calc_norm_transform(s_eigen_vectors, sigma_squared)
            self.normalisation_transforms.append(s_norm_trans)
            
            s_vectors_normalised = s_state_vector@s_norm_trans
            self.state_vectors_normalised.append(s_vectors_normalised)

        self.state_vectors = np.array(self.state_vectors)
        self.covariance_matricies = np.array(self.covariance_matricies)
        self.eigen_vectors = np.array(self.eigen_vectors)
        self.eigen_values = np.array(self.eigen_values)

    def __calc_norm_transform(self,Q,Sigma_squared):
        """
            Parameters:
                eigvecs:
                    the eigenvector matrix of the given covariance matrix
                eigvals:
                    the eigenvalue matrix of the given covariance matrix
            Returns:
                the normalisation transform = QΣ⁻¹Qᵀ
        """

        return Q @ np.linalg.inv(np.sqrt(Sigma_squared)) @ Q.T

    def __calc_state_vector(self,simulation):
            '''
                Params:
                    simulation:
                        this is the npz outputted by simsaverloader which is a dictionary containing
                        the different features of the simulation
                Returns:
                    a concatinated list (or vector) of pos_x+pos_y+vel_x+vel_y.
                    Output shape should be (T,4*n) where n is the number of boids in the simulation or readouts.
                
            '''
            x = simulation['positions']
            v = simulation['velocities']
            r_i = np.array(np.concatenate([x,v],axis=2))
            r_i = r_i.reshape(r_i.shape[0],r_i.shape[1]*r_i.shape[2])
            
            return r_i

    def __calc_covariance_matrix(self,state,assume_zero_mean:bool = True):
        '''
        Maths:
            Covariance equation:
                    $c_{xx} = \langle x_i(t) * x_j(t)\rangle$   latex version
                    c[i,j] = <xi(t) * xj(t)>                    readable version

                    Makes the assumption that xi averaged over time = 0. In other words, that xs default value is 0, and therefore, that its scale is a measure of its deviation from its norm. And therfore, the covariance, which is multiplying these two together, a combined scale, speaks to their shared deviation from the norm.

                    this is fucking varience 𝜎^2= Σ (x-x_mean)/N 
        Params:
            state:
                assumed to be shape (T,S) where T is the time steps, and s is a concatinated state space. for example, the number of boids in S would be S/4 because S should contain x and y, positions and velocities.
            
            assume_zero_mean:
                if True: the function assumes that a given state vector value has zero mean. 
                If False: the function calculates the mean for each state vector value and subtracts this avg vector from the original state vector ensuring that the state vector does in fact have zero mean for each value

        Returns:
            out:
                the symetric matrix of state S^T S averaged over time
        '''

        if not assume_zero_mean:
            #cxx = < ( x_i(t)-xi_mean) * ( xj(t)-xj_mean ) >
            state_mean = np.mean(state,axis=0) # the mean for each state value [x1_mean, ... xn_mean] (where n is the size of the state vector)
            state_zero_mean = state-state_mean # the state vector where each value has zero mean
            assert np.allclose(np.mean(state_zero_mean,axis=0),0)# ensures ts worked
            state = state_zero_mean
        
        time_steps = state.shape[0]
        state_transpose = state.T #gives (S,T)
        cxx=(state_transpose @ state)/time_steps
        
        cxx+=(10**-9)*np.eye(len(cxx)) # "To ensure numerical stability, we add a small regularization term" - lymburn et al
        return cxx

    def calc_x_covariance_matrix(self,replica_state1,replica_state2):
        '''
            cross covariance matrix
        '''
        time_steps = replica_state1.shape[0]
        css = (replica_state1.T@replica_state2)/time_steps
        return css

    def calc_consistent_capacity(self,replica1,replica2):
        css = self.calc_x_covariance_matrix(replica1,replica2)
        trace = np.trace(css)
        return trace

        # below would only work if Css is symetric. Which is only a given if replica1 = replica2.T 
        #eigen_values,_ = np.linalg.eig(css)
        #return np.sum(eigen_values)

class KernelReservoir:
    def __init__(self,kernel_number,training,testing):
        self.kernel_number = kernel_number
        '''number of observation kernels'''
        self.training = training
        '''training data set npz'''
        self.testing = testing
        '''testing data set npz'''
        self.sim_dur_T = training['time_steps'] 
        '''simulation duration'''

        self.kernels = self.generate_kernels()

    def generate_kernels(self):
        
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

    def calculate_receptive_fields(self,kernels,run,time_step):
        '''
        calculated the signal observed by each kernel at a given time step.
        '''
        x = run['positions']
        v = run['velocities']

        def kernel_function(x_i,c_m,w_m):
            ''':returns: kernal for a '''

            numer = -np.linalg.norm(x_i-c_m)**2
            denom = 2 * w_m
            return np.exp(numer/denom)

        r1,r2,r3 = [],[],[] 

        # all velocities and positions at a given time step
        velocities = v[time_step] 
        positions = x[time_step]

        for c_m, w_m in kernels:
            r1_m = 0
            r2_m = 0
            r3_m = 0

            for x_i, (vx_i, vy_i) in zip(positions, velocities):
                psi = kernel_function(x_i,c_m,w_m)

                r1_m += psi
                r2_m += psi * vx_i
                r3_m += psi * vy_i
            
            r1.append(r1_m)
            r2.append(r2_m)
            r3.append(r3_m)

        return r1,r2,r3

    def get_kernel_reservoir_state_vectorised(self, run, chunk_size=1000):

        kernels = self.kernels
        positions = run['positions']
        velocities = run['velocities']
        T = len(positions)
        
        centers = np.array([k[0] for k in kernels])
        widths = np.array([k[1] for k in kernels])
        
        r1_list, r2_list, r3_list = [], [], []
        
        for chunk_start in trange(0, T, chunk_size, desc="Vectorized"):
            chunk_end = min(chunk_start + chunk_size, T)
            
            pos_chunk = positions[chunk_start:chunk_end]
            vel_chunk = velocities[chunk_start:chunk_end]
            
            diff = pos_chunk[:, :, np.newaxis, :] - centers[np.newaxis, np.newaxis, :, :]
            distances_sq = np.sum(diff**2, axis=-1)
            psi = np.exp(-distances_sq / (2 * widths[np.newaxis, np.newaxis, :]))
            
            r1_list.append(np.sum(psi, axis=1))
            r2_list.append(np.sum(psi * vel_chunk[:, :, 0:1], axis=1))
            r3_list.append(np.sum(psi * vel_chunk[:, :, 1:2], axis=1))
        
        r1 = np.vstack(r1_list)
        r2 = np.vstack(r2_list)
        r3 = np.vstack(r3_list)
        
        return np.concatenate([r1, r2, r3], axis=1)
    
    def get_kernel_rervoir_state(self,run):
        kernels = self.kernels

        r1,r2,r3 = [],[],[]

        for t in trange(self.sim_dur_T):
            r1_t, r2_t, r3_t = self.calculate_receptive_fields(kernels,run,t)
            r1.append(r1_t)
            r2.append(r2_t)
            r3.append(r3_t)

        return np.concatenate([r1,r2,r3],axis=1)

    def ridge_prediction(self,training_reservoir_state, testing_reservoir_state, lorenz):
        lorenz_x = lorenz[:,0]
        targets_x = np.roll(lorenz_x,-1)
        self.targets_x = targets_x[:-1]

        training_reservoir_state = training_reservoir_state[:-1]

        ridge = Ridge(alpha=self.ridge_beta,fit_intercept=False)
        ridge.fit(training_reservoir_state,self.targets_x)
        
        return ridge.predict(testing_reservoir_state[:-1])

    def get_prediction(self,ridge_beta):
        self.ridge_beta = ridge_beta
        r_train = self.get_kernel_reservoir_state_vectorised(self.training)
        r_test = self.get_kernel_reservoir_state_vectorised(self.testing)

        lorenz = self.training['predator_positions']

        self.prediction = self.ridge_prediction(r_train,r_test,lorenz)

    def plot_results(self):
        if(self.prediction is None): raise Exception('No prediction has been made so cannot plot')

        lorenz_x = self.testing['predator_positions'][:,0]

        fig, ax = plt.subplots(figsize=(20, 6))
        plt.subplots_adjust(bottom=0.2)

        window_size = 2000
        max_index = len(lorenz_x)

        ax.plot(lorenz_x, color='red', label='lorenz_x')
        ax.plot(self.prediction, color='blue', linestyle='dashed', label='Prediction')

        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

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

        # Set initial view
        ax.set_xlim([0, window_size])
        plt.xlabel('time')
        plt.ylabel('lorenz_x(t+1)')
        plt.title(f'correlation coeficiant R: {self.get_correlation_coef()}')

        plt.show()

    def get_correlation_coef(self):
        numer = np.mean(self.targets_x*self.prediction,axis=0)
        denom = np.sqrt( np.mean(self.targets_x**2,axis=0) * np.mean(self.prediction**2,axis=0))
        return numer/denom
    

#path = ['tests/super_long/']
#datas = ssl.find_npzs(path)
#pa = PropertyAnalysis(datas)

#consistent_capacity = pa.calc_consistent_capacity(pa.state_vectors_normalised[0],pa.state_vectors_normalised[1])
