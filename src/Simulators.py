import numpy as np
from scipy.integrate import solve_ivp
from tqdm import trange
from scipy.spatial import KDTree
from .SaverLoader import create_mmap

EPS =1e-12 # used for avoiding divide by 0

class BoidSimulator:
    def __init__(self,parameters,use_random_seed,memory_mapping=False,chunk_size=1000):
        if len(parameters) <=0:
            raise ValueError(".ini provided is empty.")
        self.p = parameters

        # Exception to the rule that the parameters shouldn't be attributes, just because keeping the following as an attribute improves readbility so much
        self.torus = parameters['coord_system'] == 'torus'

        self.lorenz_system = LorenzSimulator(self.p['l_sigma'],self.p['l_rho'],self.p['l_beta'])
        # Randomness related params
        self.set_seed(random=use_random_seed)
        self.using_seed = use_random_seed

        # Chunking related params
        self.chunk_size = chunk_size
        self.memory_mapping = memory_mapping


    def set_seed(self,random):
        if random:
            np.random.seed(None)
        else:
            np.random.seed(self.p['random_seed'])

    ## Forces
    def __torus_distance(self,x1,x2):
        """
            Distance between two points on a torus.

            Parameters
            ----------
            x1 : vector(2)
                first point
            x2 : vector(2)
                second point

            Returns
            -------
            float
                distance between the two points
        """
        distance = x1 - x2
        if self.p['coord_system'] == 'torus':
            width = self.p['sim_width']
            distance = (distance + width/2) % width - width/2
        return distance
    
    def __repulsion_force(self,boid,neis_x):
        """ 
            boid is an np.array(2) [x,y] of a given boid
            neighoburs is an np.array(2,n) where n is the number of neighbours
        """
        if len(neis_x)==0: return np.zeros(2)

        dist = boid - neis_x if self.p['coord_system'] == 'flat' else self.__torus_distance(boid,neis_x)

        denom = (dist[:,0]**2 + dist[:,1]**2) + EPS

        return (dist / denom[:,None]).sum(axis=0)

    def __alignment_force(self,boid_v,neis_v):
        """boid is an np.array(2) [x,y] of a given boid's velocity\n neighoburs is an np.array(2,n) where n is the number of neighbours, and gives the velocities of all the neighbours"""
        force = np.sum(neis_v - boid_v, axis=0)
        return force

    def __homing_force(self,boid,home=np.array([0.0,0.0]),neis_x=None):
        if self.torus and not neis_x is None:
            width = self.p['sim_width']
            if len(neis_x)==0:
                return np.zeros(2)
            
            angles = (neis_x / width) * 2 * np.pi
            mean_cos = np.mean(np.cos(angles),axis=0)
            mean_sin = np.mean(np.sin(angles),axis=0)

            mean_angle = np.arctan2(mean_sin,mean_cos)
            
            target = ((mean_angle / (2 * np.pi)) % 1.0) * width
            diff = self.__torus_distance(target,boid)
            return diff

        return home-boid

    def __friction_force(self,boid_v):
        norm_vel = np.linalg.norm(boid_v)
        s = self.p['k_speed']
        numer = norm_vel -s
        denom = s
        return -boid_v * (numer/denom)

    def __predator_force(self,boid_x,pred_x):
        if pred_x is None: return np.array([0.0,0.0])
        
        dist = self.__torus_distance(boid_x, pred_x) if self.torus else boid_x - pred_x 
        dist_norm = np.linalg.norm(dist)

        
        # this 'if else' is the heaviside function
        if(dist_norm<=self.p['rad_predator']):
            denom = dist_norm
            numer = dist
            return (numer/denom+EPS)
        else:
            return np.array([0.0,0.0])

    def __total_force(self, boid_x, boid_v, a_neighbours, r_neighbours, h_neighbours, pred_x=None):
        no_lorenz = 1 if self.p['predator'] else 0 #if lorenz is disabled in the params
        #       |-coefficent--------------|-force-----------------|-force-params---------|
        force = ((self.p['k_alignment']*  self.__alignment_force (boid_v,a_neighbours)) +
                (self.p['k_repulsion'] *  self.__repulsion_force (boid_x,r_neighbours)) +
                (self.p['k_homing']    *  self.__homing_force    (boid_x,neis_x=h_neighbours)) +
                (self.p['k_friction']  *  self.__friction_force  (boid_v))              +
                (self.p['k_predator']  *  self.__predator_force  (boid_x,pred_x)*no_lorenz ))       

        # sigmoidal function
        force_sigmoid = self.p['alpha'] * np.tanh(self.p['beta'] * force)

        return force_sigmoid

    def __force_matrix(self,boid_xs,boid_vs,pred_x=None):
        forces = np.empty((len(boid_xs),2))

        if self.p['coord_system'] == 'flat':
            tree_points = boid_xs
            tree = KDTree(tree_points)

        elif self.p['coord_system'] == 'torus':
            tree_points = self.__torus_wrap(boid_xs)
            tree = KDTree(tree_points, boxsize=self.p['sim_width'])
            
        align_lists = tree.query_ball_point(tree_points,r=self.p['rad_alignment'])
        repul_lists = tree.query_ball_point(tree_points,r=self.p['rad_repulsion'])
        homing_lists = tree.query_ball_point(tree_points,r=self.p['rad_homing'])

        # this is iterating through each boid and applying its force
        for i, (x,v) in enumerate(zip(boid_xs,boid_vs)):
            # each agent (i) removes itself from its own list of neighbours (for each force)
            align_lists[i].remove(i)
            repul_lists[i].remove(i)
            homing_lists[i].remove(i)

            a_neis_v = boid_vs[align_lists[i]]
            r_neis_x = boid_xs[repul_lists[i]]
            h_neis_x = boid_xs[homing_lists[i]]

            forces[i] = self.__total_force(x, v, a_neis_v, r_neis_x, h_neis_x, pred_x)

        return forces

    def generate_flock(self,flock_size,spawn_bounds,random_velocity=False,random_position=False):
        #the reason for it being done as follows below is to protect against cases where the tuple orders the min and max lim differently
        spawn_min = min(spawn_bounds)
        spawn_max = max(spawn_bounds)
        spawn_width = np.abs(spawn_max-spawn_min)

        self.set_seed(random_position) #undo the random seed (may not do anything if already not random)
        if self.p['coord_system']=='flat':
            x = spawn_min + spawn_width * np.random.beta(2, 2, size=(flock_size, 2))
        elif self.torus:
            x = self.p['sim_width']/2+spawn_min + spawn_width * np.random.beta(2, 2, size=(flock_size, 2))
        self.set_seed(self.using_seed) #reapply the random seed (if its being used)

        self.set_seed(random_velocity) #undo the random seed (may not do anything if already not random)
        v=np.random.rand(flock_size,2)
        self.set_seed(self.using_seed) #reapply the random seed (if its being used)

        # returns positions, velocities, neighbour data
        return x, v

    def __torus_wrap(self,positions):
        '''
            assumes to take same positions in shape (n,2) (where n is the number of things with a position)
        '''
        #python modulo wraps negative numbers in the way one expects
        wrapped = positions%self.p['sim_width']
        return wrapped

    def __physics_step(self,current_pos,current_vel,prior_lorenz_x):
        '''
        the priors are the given parameter at t
        '''
        current_x = current_pos.copy()
        current_v = current_vel.copy()

        # Calculate the force matrix (forces acting on each boid) at the current step
        fm = self.__force_matrix(current_x,current_v,prior_lorenz_x)

        # Update velocity and position matrix
        new_v = current_v + (fm * self.p['delta_t'])
        #new_x = current_x + (current_v * self.p['delta_t']) This is what Lymburn does but my superviser and I agree it's probably supposed to be done like the line below
        new_x = current_x + (new_v * self.p['delta_t'])

        return new_v,new_x

    def run_simulation(self):
        boid_count = self.p['boid_count']
        simulation_steps = self.p['simulation_steps']
        mode_shape = (simulation_steps,boid_count,2)

        if self.memory_mapping:
            # times 2, for two positions, and two velocities
            positions,_ = create_mmap('boid_positions_',mode_shape)
            velocities,_ = create_mmap('boid_velocities_',mode_shape)
        else:
            positions = np.zeros(mode_shape)
            velocities= np.zeros(mode_shape)

        self.spawn_bounds = (self.p['spawn_min'],self.p['spawn_max'])

        lorenz = self.lorenz_system.generate_lorenz(simulation_steps, self.p['l_sampling_rate'], self.p['x_lorenz'], self.p['y_lorenz'], self.p['z_lorenz'])
        p, v = self.generate_flock(boid_count, self.spawn_bounds, self.p['random_velocity'], self.p['random_position'])

        if self.torus:
            p = self.__torus_wrap(p) # wrap each boid pos over 200 boids
            #1. Center lorenz around a center where 
            lorenz = lorenz+self.p['sim_width']/2
            #2. Wrap the recentered lorenz
            lorenz = self.__torus_wrap(lorenz)# wrap each coordinate over all time steps
        positions[0] = p
        velocities[0] = v

        #-1 because the frist step is them spawning. indexed 0 shown above, setting the inital P and V
        for t in trange(simulation_steps - 1, desc="Simulation",position=1,leave=True):
            current_pos = positions[t]
            current_vel = velocities[t]

            new_v, new_x = self.__physics_step(current_pos, current_vel, lorenz[t])

            if self.p['coord_system'] == 'torus':
                new_x = self.__torus_wrap(new_x)

            positions[t + 1] = new_x
            velocities[t + 1] = new_v

            if self.memory_mapping and (t + 1) % self.chunk_size == 0:
                positions.flush()
                velocities.flush()

        return {
            "positions": positions,
            "velocities": velocities,
            "predator_positions": lorenz,
            "simulation_steps" :simulation_steps,
            "boid_count" : boid_count,
            "bounds": self.spawn_bounds,
            "config":self.p
        }

    @staticmethod
    def construct_view_dict(datas:list):
        """
        This function is used for converting the data output of my simulator into the more standardised input required by functions in BoidVisualiser.py
        In other words, the function exists for compatability sake.
        Parameters
        ----------
        datas : list
            list of data outputs from the class
        """
        replicas_for_visualiser = []
        for d in datas:
            pos = d['positions']
            vel = d['velocities']
            signal = d['predator_positions']
            replica_dict = {'positions':pos,'velocities':vel,'input_signal':signal}
            replicas_for_visualiser.append(replica_dict)
        return replicas_for_visualiser


class LorenzSimulator:
    def __init__(self,sigma,rho,beta):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def lorenz_equations(self,t,start_states):
        x,y,z = start_states
        dxBYdt = self.sigma * (y-x)
        dyBYdt = x * (self.rho - z) - y
        dzBYdt = (x * y) - (self.beta * z)
        return dxBYdt,dyBYdt,dzBYdt

    def generate_lorenz(self,simulation_steps, sample_rate, x_init, y_init, z_init,norm_std=2):
        t = np.arange(simulation_steps, dtype=float) * sample_rate

        soln = solve_ivp(self.lorenz_equations, t_span=(t[0],t[-1]) ,y0=(x_init,y_init,z_init) ,dense_output=True)
        coords = soln.sol(t).T

        if norm_std is None:#dont rescale in this case
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
        else:
            # Lambda for rescaling the output to have std of 2 and mean 0 (as specified in Lymburn et al)
            rescale = lambda axis: norm_std * (axis - np.mean(axis)) / np.std(axis)
            x_coords = rescale(coords[:, 0])
            y_coords = rescale(coords[:, 1])

        lorenz_series = np.column_stack((x_coords,y_coords))
        return lorenz_series