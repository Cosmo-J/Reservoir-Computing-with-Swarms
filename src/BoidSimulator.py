import numpy as np
from scipy.integrate import solve_ivp
from tqdm import trange
from scipy.spatial import KDTree
from .SaverLoader import create_mmap

EPS =1e-12 # used for avoiding divide by 0

class BoidSimulator:
    def __init__(self,parameters,use_seed,memory_mapping=False,chunk_size=1000):
        if len(parameters) <=0:
            raise ValueError(".ini provided is empty.")

        self.p = parameters

        # Randomness related params
        self.set_seed(random=use_seed)
        self.using_seed = use_seed

        # Chunking related params
        self.chunk_size = chunk_size
        self.memory_mapping = memory_mapping


    def set_seed(self,random):
        if random:
            np.random.seed(None)
        else:
            np.random.seed(self.p['RANDOM_SEED'])

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
        if self.p['COORD_SYSTEM']=='torus' and not neis_x is None:
            width = self.p['SIM_WIDTH']
            if len(neis_x)==0:
                return np.zeros(2)
            
            angles = (neis_x / self.p['SIM_WIDTH']) * 2 * np.pi
            mean_cos = np.mean(np.cos(angles),axis=0)
            mean_sin = np.mean(np.sin(angles),axis=0)

            mean_angle = np.arctan2(mean_sin,mean_cos)
            
            target = ((mean_angle / (2 * np.pi)) % 1.0) * self.p['SIM_WIDTH']
            diff = target - boid
            diff = (diff + width/2) % width - width/2
            return diff

        return home-boid

    def __friction_force(self,boid_v):
        speed = np.hypot(boid_v[0],boid_v[1])
        return -boid_v * ((speed - self.p['K_SPEED']) / self.p['K_SPEED'])

    def __predator_force(self,boid_x,pred_x):
        if pred_x is None: return np.array([0.0,0.0])
        d= np.linalg.norm(boid_x - pred_x)

        
        # this 'if else' is the heaviside function
        if(d<=self.p['RAD_PREDATOR']):
            denom = d**2
            numer = boid_x-pred_x
            return (numer/denom+EPS)
        else:
            return np.array([0.0,0.0])

    def __total_force(self, boid_x, boid_v, a_neighbours, r_neighbours, h_neighbours, pred_x=None):

        no_lorenz = 1 if self.p['PREDATOR'] else 0 #if lorenz is disabled in the params
    #           |-coefficent--------------|-force-----------------|-force-params---------|
        force = ((self.p['K_ALIGNMENT']*  self.__alignment_force (boid_v,a_neighbours)) +
                (self.p['K_REPULSION'] *  self.__repulsion_force (boid_x,r_neighbours)) +
                (self.p['K_HOMING']    *  self.__homing_force    (boid_x,neis_x=h_neighbours)) +
                (self.p['K_FRICTION']  *  self.__friction_force  (boid_v))              +
                (self.p['K_PREDATOR']  *  self.__predator_force  (boid_x,pred_x)*no_lorenz ))       

        # sigmoidal function
        force_sigmoid = self.p['ALPHA'] * np.tanh(self.p['BETA'] * force)

        return force_sigmoid

    def __force_matrix(self,boid_xs,boid_vs,pred_x=None):
        forces = np.empty((len(boid_xs),2))

        if self.p['COORD_SYSTEM'] == 'flat':
            tree_points = boid_xs
            tree = KDTree(tree_points)

        elif self.p['COORD_SYSTEM'] == 'torus':
            tree_points = self.__lorenz_wrap(boid_xs)
            tree = KDTree(tree_points, boxsize=self.p['SIM_WIDTH'])
            
        align_lists = tree.query_ball_point(tree_points,r=self.p['RAD_ALIGNMENT'])
        repul_lists = tree.query_ball_point(tree_points,r=self.p['RAD_REPULSION'])
        homing_lists = tree.query_ball_point(tree_points,r=self.p['RAD_HOMING'])

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

    def __lorenz_equations(self,t,start_states):
        x,y,z = start_states
        dxBYdt = self.p['L_SIGMA'] * (y-x)
        dyBYdt = x * (self.p['L_RHO'] - z) - y
        dzBYdt = (x * y) - (self.p['L_BETA'] * z)
        return dxBYdt,dyBYdt,dzBYdt

    def generate_lorenz(self,simulation_steps, sample_rate, x_init, y_init, z_init,norm_std=2):
        # Lambda for rescaling the output to have std of 2 and mean 0 (as specified in Lymburn et al)
        rescale = lambda axis: norm_std * (axis - np.mean(axis)) / np.std(axis)

        t = np.arange(simulation_steps, dtype=float) * sample_rate

        soln = solve_ivp(self.__lorenz_equations, t_span=(t[0],t[-1]) ,y0=(x_init,y_init,z_init) ,dense_output=True)
        coords = soln.sol(t).T

        rescaled_x_coords = rescale(coords[:, 0])
        rescaled_y_coords = rescale(coords[:, 1])
        
        lorenz_series = np.column_stack((rescaled_x_coords,rescaled_y_coords))

        return lorenz_series


    def generate_flock(self,flock_size,spawn_bounds,random_velocity=False,random_position=False):
        #the reason for it being done as follows below is to protect against cases where the tuple orders the min and max lim differently
        spawn_min = min(spawn_bounds)
        spawn_max = max(spawn_bounds)
        spawn_width = np.abs(spawn_max-spawn_min)

        self.set_seed(random_position) #undo the random seed (may not do anything if already not random)
        if self.p['COORD_SYSTEM']=='flat':
            x = spawn_min + spawn_width * np.random.beta(2, 2, size=(flock_size, 2))
        elif self.p['COORD_SYSTEM']=='torus':
            x = self.p['SIM_WIDTH']/2+spawn_min + spawn_width * np.random.beta(2, 2, size=(flock_size, 2))
        self.set_seed(self.using_seed) #reapply the random seed (if its being used)

        self.set_seed(random_velocity) #undo the random seed (may not do anything if already not random)
        v=np.random.rand(flock_size,2)
        self.set_seed(self.using_seed) #reapply the random seed (if its being used)

        # returns positions, velocities, neighbour data
        return x, v

    def __lorenz_wrap(self,positions):
        '''
            assumes to take same positions in shape (n,2) (where n is the number of things with a position)
        '''
        #python modulo wraps negative numbers in the way one expects
        wrapped = positions%self.p['SIM_WIDTH']
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
        new_v = current_v + (fm * self.p['DELTA_T'])
        #new_x = current_x + (current_v * self.p['DELTA_T']) This is what Lymburn does but my superviser and I agree it's probably supposed to be done like the line below
        new_x = current_x + (new_v * self.p['DELTA_T'])

        return new_v,new_x

    def run_simulation(self):
        boid_count = self.p['BOID_COUNT']
        simulation_steps = self.p['SIMULATION_STEPS']
        mode_shape = (simulation_steps,boid_count,2)

        if self.memory_mapping:
            # times 2, for two positions, and two velocities
            positions,_ = create_mmap('boid_positions_',mode_shape)
            velocities,_ = create_mmap('boid_velocities_',mode_shape)
        else:
            positions = np.zeros(mode_shape)
            velocities= np.zeros(mode_shape)

        self.spawn_bounds = (self.p['SPAWN_MIN'],self.p['SPAWN_MAX'])

        lorenz = self.generate_lorenz(simulation_steps, self.p['L_SAMPLING_RATE'], self.p['X_LORENZ'], self.p['Y_LORENZ'], self.p['Z_LORENZ'])
        p, v = self.generate_flock(boid_count, self.spawn_bounds, self.p['RANDOM_VELOCITY'], self.p['RANDOM_POSITION'])

        if self.p['COORD_SYSTEM']=='torus':
            p = self.__lorenz_wrap(p) # wrap each boid pos over 200 boids

            #1. Center lorenz around a center where 
            lorenz = lorenz+self.p['SIM_WIDTH']/2
            #2. Wrap the recentered lorenz
            lorenz = self.__lorenz_wrap(lorenz)# wrap each coordinate over 1000 steps
        
        positions[0] = p
        velocities[0] = v

        #-1 because the frist step is them spawning. indexed 0 shown above, setting the inital P and V
        for t in trange(simulation_steps - 1, desc="Simulation",position=1,leave=True):
            current_pos = positions[t]
            current_vel = velocities[t]

            new_v, new_x = self.__physics_step(current_pos, current_vel, lorenz[t])

            if self.p['COORD_SYSTEM'] == 'torus':
                new_x = self.__lorenz_wrap(new_x)

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

