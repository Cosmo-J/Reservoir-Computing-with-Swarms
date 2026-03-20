import numpy as np
from scipy.integrate import solve_ivp
from tqdm import trange
from scipy.spatial import KDTree
import threading

from concurrent.futures import ThreadPoolExecutor as TPE, as_completed

from .SimSaverLoader import SimSaverLoader
from .SimParams import SimParams


EPS =1e-12 # used for avoiding divide by 0


class BoidSimulator:
    def initialise(self,param_loader:SimParams= None, sim_saver_loader:SimSaverLoader=None):
        '''
            sets and or creates a SimParms and SimSaverLoader object for the BoidSimulator
        '''
        if param_loader is None:
            param_loader = SimParams()
        if sim_saver_loader is None:
            sim_saver_loader = SimSaverLoader()

        self.param_loader = param_loader
        self.saver_loader = sim_saver_loader

        return param_loader,sim_saver_loader

    def PARAMS(self):
        return self.param_loader.params

    def set_seed(self,seed):
        if isinstance(seed,int): 
            np.random.seed(seed)
            print(f"Using custom np.random seed: {seed}")
        else: 
            np.random.seed(1)
        
## Forces
    def __repulsion_force(self,boid,neis_x):
        """ 
            boid is an np.array(2) [x,y] of a given boid
            neighoburs is an np.array(2,n) where n is the number of neighbours
        """
        if len(neis_x)==0: return np.zeros(2)

        dist = boid - neis_x

        denom = (dist[:,0]**2 + dist[:,1]**2) + EPS

        return (dist / denom[:,None]).sum(axis=0)

    def __alignment_force(self,boid_v,neis_v):
        """boid is an np.array(2) [x,y] of a given boid's velocity\n neighoburs is an np.array(2,n) where n is the number of neighbours, and gives the velocities of all the neighbours"""
        force = np.array([0.0,0.0])
        for n in neis_v:
            force+= n - boid_v
        return force

    def __homing_force(self,boid,home=np.array([0.0,0.0]),neis_x=None):
        if self.PARAMS()['COORD_SYSTEM']=='torus':
            width = self.PARAMS()['SIM_WIDTH']
            if len(neis_x)==0:
                return np.zeros(2)
            
            angles = (neis_x / self.PARAMS()['SIM_WIDTH']) * 2 * np.pi
            mean_cos = np.mean(np.cos(angles),axis=0)
            mean_sin = np.mean(np.sin(angles),axis=0)

            mean_angle = np.arctan2(mean_sin,mean_cos)
            
            target = ((mean_angle / (2 * np.pi)) % 1.0) * self.PARAMS()['SIM_WIDTH']
            diff = target - boid
            diff = (diff + width/2) % width - width/2
            return diff

        return home-boid

    def __friction_force(self,boid_v):
        speed = np.hypot(boid_v[0],boid_v[1])
        return -boid_v * ((speed - self.PARAMS()['K_SPEED']) / self.PARAMS()['K_SPEED'])

    def __predator_force(self,boid_x,pred_x):
        if pred_x is None: return np.array([0.0,0.0])
        d= np.linalg.norm(boid_x - pred_x)

        
        # this 'if else' is the heaviside function
        if(d<=self.PARAMS()['RAD_PREDATOR']):
            denom = d**2
            numer = boid_x-pred_x
            return (numer/denom+EPS)
        else:
            return np.array([0.0,0.0])

    def __total_force(self, boid_x, boid_v, a_neighbours, r_neighbours, h_neighbours, pred_x=None):

        no_lorenz = 1 if self.PARAMS()['PREDATOR'] else 0 #if lorenz is disabled in the params
    #           |-coefficent-----------------|-force--------------------|-force-params---------|
        force = ((self.PARAMS()['K_ALIGNMENT']*  self.__alignment_force (boid_v,a_neighbours)) +
                (self.PARAMS()['K_REPULSION'] *  self.__repulsion_force (boid_x,r_neighbours)) +
                (self.PARAMS()['K_HOMING']    *  self.__homing_force    (boid_x,neis_x=h_neighbours)) +
                (self.PARAMS()['K_FRICTION']  *  self.__friction_force  (boid_v))              +
                (self.PARAMS()['K_PREDATOR']  *  self.__predator_force  (boid_x,pred_x)*no_lorenz ))       

        # sigmoidal function
        force_sigmoid = self.PARAMS()['ALPHA'] * np.tanh(self.PARAMS()['BETA'] * force)

        return force_sigmoid

    def __force_matrix(self,boid_xs,boid_vs,pred_x=None):
        forces = np.empty((len(boid_xs),2))

        if self.PARAMS()['COORD_SYSTEM'] == 'flat':
            tree_points = boid_xs
            tree = KDTree(tree_points)

        elif self.PARAMS()['COORD_SYSTEM'] == 'torus':
            tree_points = self.__lorenz_wrap(boid_xs)
            tree = KDTree(tree_points, boxsize=self.PARAMS()['SIM_WIDTH'])
            
        align_lists = tree.query_ball_point(tree_points,r=self.PARAMS()['RAD_ALIGNMENT'])
        repul_lists = tree.query_ball_point(tree_points,r=self.PARAMS()['RAD_REPULSION'])
        homing_lists = tree.query_ball_point(tree_points,r=self.PARAMS()['RAD_HOMING'])

        # this is iterating through each boid and applying its force
        for i, (x,v) in enumerate(zip(boid_xs,boid_vs)):
            # remove self from the ids in the neighbour lists
            align_lists[i].remove(i)
            repul_lists[i].remove(i)
            homing_lists[i].remove(i)

            a_neis_v = boid_vs[align_lists[i]]
            r_neis_x = boid_xs[repul_lists[i]]
            h_neis_x = boid_xs[homing_lists[i]]

            forces[i] = self.__total_force( boid_x = x,
                                            boid_v=v,
                                            a_neighbours=a_neis_v,
                                            r_neighbours=r_neis_x,
                                            h_neighbours=h_neis_x,
                                            pred_x=pred_x)


        return forces

    def __lorenz_equations(self,t,start_states):
        x,y,z = start_states
        dxBYdt = self.PARAMS()['L_SIGMA'] * (y-x)
        dyBYdt = x * (self.PARAMS()['L_RHO'] - z) - y
        dzBYdt = (x * y) - (self.PARAMS()['L_BETA'] * z)
        return dxBYdt,dyBYdt,dzBYdt

    def __generate_lorenz(self,time_steps, sample_rate, x_init, y_init, z_init):
        rescale = lambda axis: 2 * (axis - np.mean(axis)) / np.std(axis)
        lorenz_segment = time_steps*sample_rate

        soln = solve_ivp(self.__lorenz_equations, t_span=(0,lorenz_segment) ,y0=(x_init,y_init,z_init) ,dense_output=True)
        t = np.linspace(0, lorenz_segment, time_steps)
        coords = soln.sol(t).T

        rescaled_x_coords = rescale(coords[:, 0])
        rescaled_y_coords = rescale(coords[:, 1])
        
        lorenz_series = np.column_stack((rescaled_x_coords,rescaled_y_coords))

        return lorenz_series

    def __generate_flock(self,flock_size,spawn_bounds,random_velocity=False):
        #the reason for it being done as follows below is to protect against cases where the tuple orders the min and max lim differently
        spawn_min = min(spawn_bounds)
        spawn_max = max(spawn_bounds)
        spawn_width = np.abs(spawn_max-spawn_min)

        if self.PARAMS()['COORD_SYSTEM']=='flat':
            x = spawn_min + spawn_width * np.random.beta(2, 2, size=(flock_size, 2))
        elif self.PARAMS()['COORD_SYSTEM']=='torus':
            x = self.PARAMS()['SIM_WIDTH']/2+spawn_min + spawn_width * np.random.beta(2, 2, size=(flock_size, 2))

        if random_velocity: 
            v=np.random.rand(flock_size,2)
        else: 
            v = np.zeros((flock_size,2),dtype=float)

        # returns positions, velocities, neighbour data
        return x, v

    def __lorenz_wrap(self,positions):
        '''
            assumes to take same positions in shape (n,2) (where n is the number of things with a position)
        '''
        #python modulo wraps negative numbers in the way one expects
        wrapped = positions%self.PARAMS()['SIM_WIDTH']
        return wrapped

    def __physics_step(self,t,positions,velocities,prior_lorenz_x):
        '''
        the priors are the given parameter at t
        '''
        current_x = positions[t]
        current_v = velocities[t]

        #2. Calculate the force matrix (forces acting on each boid) at the current step
        fm = self.__force_matrix(current_x,current_v,prior_lorenz_x)

        #3. update velocity matrix
        #4. update position matrix
        new_v = current_v + (fm * self.PARAMS()['DELTA_T'])
        new_x = current_x + (new_v * self.PARAMS()['DELTA_T'])

        return new_v,new_x

    def run_simulation(self):
        if len(self.PARAMS()) is None:
            raise Exception('Please generate params and apply them before running simulation. apply_params()')

        positions = []
        velocities = []

        self.spawn_bounds = (self.PARAMS()['SPAWN_MIN'],self.PARAMS()['SPAWN_MAX'])

        lorenz = self.__generate_lorenz(self.PARAMS()['TIME_STEPS'], self.PARAMS()['L_SAMPLING_RATE'], self.PARAMS()['X_LORENZ'], self.PARAMS()['Y_LORENZ'], self.PARAMS()['Z_LORENZ'])
        p, v = self.__generate_flock(self.PARAMS()['BOID_COUNT'], self.spawn_bounds, self.PARAMS()['RANDOM_VELOCITY'])

        if self.PARAMS()['COORD_SYSTEM']=='torus':
            p = self.__lorenz_wrap(p) # wrap each boid pos over 200 boids

            #1. Center lorenz around a center where 
            lorenz = lorenz+self.PARAMS()['SIM_WIDTH']/2
            #2. Wrap the recentered lorenz
            lorenz = self.__lorenz_wrap(lorenz)# wrap each coordinate over 1000 steps
        positions.append(p)
        velocities.append(v)


        """below is a very excentric way of getting the number from the end of the thread name seen <Thread(ThreadPoolExecutor-0_0, started 6119583744)> (which is what current_thread() returns in a MT scenario)
        otherwise rely on the failure to make the letter d an int to state that its a single threading scenario and thread indent should be 0 LOL
        not too worried about the bad practise here considering the code is purely for aesthetics"""
        try:
            thread_indent = int(threading.current_thread().name[-1])+1
        except:
            thread_indent=0

        for t in trange(self.PARAMS()['TIME_STEPS'] - 1,desc=f"Thread: {threading.current_thread().name}",position=thread_indent,leave=False):
            new_v, new_x = self.__physics_step(t,positions, velocities, lorenz[t])

            #wrap the new positions
            if self.PARAMS()['COORD_SYSTEM']=='torus':
                new_x = self.__lorenz_wrap(new_x)
            
            positions.append(new_x)
            velocities.append(new_v)


        return {
            "positions": positions,
            "velocities": velocities,
            "predator_positions": lorenz,
            "bounds": self.spawn_bounds,
            "coord_type":self.PARAMS()['COORD_SYSTEM'],
            "boid_count": self.PARAMS()['BOID_COUNT'],
            "time_steps": self.PARAMS()['TIME_STEPS'],
            "sim_width":self.PARAMS()['SIM_WIDTH'],
            "config":self.PARAMS()
        }

