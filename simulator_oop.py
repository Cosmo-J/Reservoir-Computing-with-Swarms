#this is a copy which attempts to use an external params manager
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import trange
from tqdm import tqdm
from datetime import datetime
import argparse
import os
import configparser
from viewer import View as SimViewer
from scipy.spatial import cKDTree
import threading
from concurrent.futures import ThreadPoolExecutor as TPE, as_completed


parser = argparse.ArgumentParser()
parser.add_argument('--run',            '-r',   type=str,   nargs='?', const=True,  help="Run a new simulation given a .ini file path of parameters.")
parser.add_argument('--iterations',     '-i',   type=int,   nargs='?', default=1,   help="Use with run or runview to perform multiple simulations.")
parser.add_argument('--seed',                   type=int,   nargs='?', const=1,     help="Give a random seed to produce identical runs. Default seed is '1'.")
parser.add_argument('--generate-params','-g',   type=str,   nargs='?', const='./',   help="generate an empty .ini with default parameters.")
parser.add_argument('--view',           '-v',   type=str,   nargs='+',              help="View a previous simulation, given the file path of a valid '.npz'.")
parser.add_argument('--save',           '-s',   type=str,   nargs='?', const=False,  help="If iterations is more than 1, this parameter is taken as the save directory. Otherwise")
parser.add_argument('--multithread',    '-mt',   type=bool, nargs='?', default=False,  const=True, help="set true ")

# Optimisation focused definition
EMPTY_0x2 = np.empty((0, 2))
EPS =1e-12 # used for avoiding divide by 0
DEFAULT_SAVE_PATH = 'tests/boid_runs'
PARAMS = {}

class SimSaverLoader:
    def __init__(self,save_path='boid_runs',config_title = 'NaN'):
        self.save_dir = save_path
        self.config_title = config_title

    def _unique_name(self,candidate_name:str):
        '''
        :candidate_name: this is the potential file name /path/to/file
        :return: Modified path /path/to/file_run_12.npz
        '''
        extension = '.npz'
        iter = ''
        candidate_path = f'{self.save_dir}/{candidate_name}_run'
        while os.path.exists(candidate_path+str(iter)+extension):
            if isinstance(iter,str): 
                # this is for the first loop only
                iter = 0
            else: 
                iter+=1

            if iter-10 > len(os.listdir(self.save_dir)):
                raise Exception("Couldn't find a valid name when saving the file. Iterations exceeded number of files in the directory.")
            
        return candidate_path+str(iter)+extension


    def save_run(self, data, name= None,config_title:str = None) -> str:
        os.makedirs(self.save_dir, exist_ok=True)
        if not name:
            name = os.path.dirname(self.save_dir)
        
        full_save_path_and_name = self._unique_name(name) 
        
        np.savez(
            full_save_path_and_name,
            positions=data["positions"],
            velocities=data["velocities"],
            predator_positions=data["predator_positions"],
            time_steps=data["time_steps"],
            boid_count=data["boid_count"],
            bounds=data["bounds"],
            config=PARAMS,
            config_title = [config_title]
        )

        print(f"we be saving - {full_save_path_and_name}")
        return full_save_path_and_name

    def set_save_dir(self, path:str):
        """
            if the path is directory, name the runs the smallest number iter inside said directory
        """
        if os.path.isdir(path):
            self.save_dir = path
        else:
            raise FileNotFoundError(f"Directory {path} doesn't exist or cant be found.")
                    


    def find_npzs(self, paths:str):
        '''
        Takes some paths, of directories (in which it searches for npzs, or npz paths)
        :return: array of dictionaries containing the npz runs it found
        '''
        npzs = []
        for p in paths:
            if os.path.isdir(p):
                sub_files = os.scandir(p)
                npzs.extend([f.path for f in sub_files if '.npz' in f.name])
            else:
                npzs.append(p)
        datas =[self.load_run(f) for f in npzs]
        print(f'Loaded {len(datas)} run(s)')

        return datas

    def load_run(self, path: str) -> dict:
        z = np.load(path, allow_pickle=True)
        run_dict = {}
        run_dict["positions"] = z.get("positions"),
        run_dict["velocities"] = z.get("velocities"),
        run_dict["predator_positions"] = z.get("predator_positions"),
        run_dict["time_steps"] = int(z.get("time_steps")),
        run_dict["boid_count"] = int(z.get("boid_count")),
        run_dict["bounds"] = z.get("bounds"),
        run_dict["config_title"] = z.get("config_title")
        
        #print(f"Checking if features of '{path}' are up to date...")
        for k,v in run_dict.items():
            if type(v) is tuple:
                run_dict[k] = v[0]
            if v is None:
                print(f"\tvalue for '{k}' is missing, features relating to this will not work thus.")

        return run_dict


class SimParams:
    def __init__(self):
        self.DEFAULTS = {
            # Simulation params
            "sim":
                {
                    "DELTA_T": 0.02,
                    "TIME_STEPS": 100,
                    "BOID_COUNT": 200,
                    "SPAWN_MIN": -1.0,
                    "SPAWN_MAX": 1.0,
                    "RANDOM_VELOCITY": False
                },

            # force constants
            "forces":
                {
                    "K_SPEED" : 10.0,
                    "K_REPULSION" : 1,
                    "K_ALIGNMENT" : 0.1,
                    "K_HOMING" : 2.0,
                    "K_FRICTION" : 20.0,
                    "K_PREDATOR"  : 100
                },

            # sigmoidal function
            "sigmoid":
                {
                    "BETA" : 0.1,
                    "ALPHA" : 200.0
                },

            # neighbour radii
            "radii":
                {
                    "RAD_ALIGNMENT" : 1,
                    "RAD_REPULSION" : 1,
                    "RAD_PREDATOR" : 1
                },

            # Lorenz conditions
            "lorenz":
                {
                    "L_SIGMA" : 10.0,
                    "L_RHO" : 28.0,
                    "L_BETA" : 8/3,
                    "X_LORENZ" : 0.0,
                    "Y_LORENZ" : 1.0,
                    "Z_LORENZ" : 1.05,
                    "L_SAMPLING_RATE" : 0.02 # number of sample per time step, also kind of predator speed
                }
        }
        self.params = self.DEFAULTS

    
    def get_flat_params(self):
        flat = {}
        for section_title,section in self.DEFAULTS.items():
            for k,v in section.items():
                flat[k] = v
        return flat
    
    def load_params_from_ini(self,path):
        # some sort of path validation to ensure that
        '''
            1. path exists
            2. ini exists
            3. ini is up to date
                if not, add the new rows and give them the default values
        '''


        cfg = configparser.ConfigParser()
        cfg.read(path)
        self.params = {}
        for section_title,section in self.DEFAULTS.items():
            for k,v in section.items():
                cast = type(v)
                if cfg.has_option(section_title,k):
                    self.params[k] = cast(cfg.get(section_title,k))
                else:
                    raise Exception(f'Config {path} is missing a key:value for {section_title} {k}\n\tPlease update the .ini to use this config file')
        
        return self.params
    
    def write_default_ini(self,path):
        cfg = configparser.ConfigParser()
        for section_title,section in self.DEFAULTS.items():
            cfg[section_title] = {}
            for k,v in section.items():
                cfg[section_title][k] = str(v)
        
        
        with open(path,'w',encoding="utf-8") as f:
            cfg.write(f)

class BoidSimulator:
    def __init__(self):
        pass

    def initialise(self,param_loader:SimParams= None, sim_saver_loader:SimSaverLoader=None):
        '''
            sets and or creates a SimParms and SimSaverLoader object for the BoidSimulator
        '''
        if param_loader is None:
            param_loader = SimParams()
        if sim_saver_loader is None:
            sim_saver_loader = SimSaverLoader()

        self.param_loader = param_loader
        self.PARAMS = param_loader.get_flat_params()
        self.saver_loader = sim_saver_loader

        return param_loader,sim_saver_loader


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

        d = boid - neis_x
        denom = (d[:,0]**2 + d[:,1]**2) + EPS
        return (d / denom[:,None]).sum(axis=0)

    def __alignment_force(self,boid_v,neis_v):
        """boid is an np.array(2) [x,y] of a given boid's velocity\n neighoburs is an np.array(2,n) where n is the number of neighbours, and gives the velocities of all the neighbours"""
        force = np.array([0.0,0.0])
        for n in neis_v:
            force+= n - boid_v
        return force

    def __homing_force(self,boid,home=np.array([0.0,0.0])):
        return home-boid

    def __friction_force(self,boid_v):
        speed = np.hypot(boid_v[0],boid_v[1])
        return -boid_v * ((speed - self.PARAMS['K_SPEED']) / self.PARAMS['K_SPEED'])

    def __predator_force(self,boid_x,pred_x):
        if pred_x is None: return np.array([0.0,0.0])
        d= np.linalg.norm(boid_x - pred_x)

        # this 'if else' is the heaviside function
        if(d<=self.PARAMS['RAD_PREDATOR']):
            numer = boid_x-pred_x
            denom = d**2
            return (numer/denom)
        else:
            return np.array([0.0,0.0])

    def __total_force(self,boid_x,boid_v,a_neighbours,r_neighbours,pred_x=None):
    #           |-coefficent-----------------|-force---------------|-force-params---------|
        force = ((self.PARAMS['K_ALIGNMENT']*  self.__alignment_force (boid_v,a_neighbours)) +
                (self.PARAMS['K_REPULSION'] *  self.__repulsion_force (boid_x,r_neighbours)) +
                (self.PARAMS['K_FRICTION']  *  self.__friction_force  (boid_v)) +
                (self.PARAMS['K_HOMING']    *  self.__homing_force    (boid_x)) +
                (self.PARAMS['K_PREDATOR']  *  self.__predator_force  (boid_x,pred_x) ))       

        # sigmoidal function
        force_sigmoid = self.PARAMS['ALPHA'] * np.tanh(self.PARAMS['BETA'] * force)

        return force_sigmoid

    def __force_matrix(self,boid_xs,boid_vs,pred_x=None):
        forces = np.empty((len(boid_xs),2))

        tree = cKDTree(boid_xs)
        align_lists = tree.query_ball_point(boid_xs,r=self.PARAMS['RAD_ALIGNMENT'])
        repulsion_lists = tree.query_ball_point(boid_xs,r=self.PARAMS['RAD_REPULSION'])

        for i, (x,v) in enumerate(zip(boid_xs,boid_vs)):
            a_idx = [j for j in align_lists[i] if j != i]
            r_idx = [j for j in repulsion_lists[i] if j != i]

            a_neis_v = boid_vs[a_idx] if a_idx else EMPTY_0x2
            r_neis_x = boid_xs[r_idx] if r_idx else EMPTY_0x2

            forces[i] = self.__total_force(x,v,a_neis_v,r_neis_x,pred_x)


        return forces

    def __lorenz_equations(self,t,start_states):
        x,y,z = start_states
        dxBYdt = self.PARAMS['L_SIGMA'] * (y-x)
        dyBYdt = x * (self.PARAMS['L_RHO'] - z) - y
        dzBYdt = (x * y) - (self.PARAMS['L_BETA'] * z)
        return dxBYdt,dyBYdt,dzBYdt

    def __generate_lorenz(self,time_steps, sample_rate, x_init, y_init, z_init):
        rescale = lambda axis: 2 * (axis - np.mean(axis)) / np.std(axis)

        lorenz_segment = time_steps*sample_rate

        soln = solve_ivp(self.__lorenz_equations, t_span=(0,lorenz_segment) ,y0=(x_init,y_init,z_init) ,dense_output=True)
        t = np.linspace(0, lorenz_segment, time_steps)
        coords = soln.sol(t).T

        rescaled_x_coords = rescale(coords[:, 0])
        rescaled_y_coords = rescale(coords[:, 1])
        return np.column_stack((rescaled_x_coords,rescaled_y_coords))

    def __generate_flock(self,flock_size,lim,random_velocity=False):
        #x = np.random.uniform(lim[0], lim[1], size=(flock_size, 2))
        x = lim[0] + (lim[1] - lim[0]) * np.random.beta(2, 2, size=(flock_size, 2))

        if random_velocity: 
            v=np.random.rand(flock_size,2)
        else: 
            v = np.zeros((flock_size,2),dtype=float)

        # returns positions, velocities, neighbour data
        return x, v

    def __evolve(self,t,positions,velocities,prior_lorenz_x):
        '''
        the priors are the given parameter at t
        '''
        current_x = positions[t]
        current_v = velocities[t]

        #2. Create force matrix
        fm = self.__force_matrix(current_x,current_v,prior_lorenz_x)

        #3. update velocity matrix
        #4. update position matrix
        new_v = current_v + (fm * self.PARAMS['DELTA_T'])
        new_x = current_x + (new_v * self.PARAMS['DELTA_T'])

        return new_v,new_x

    def run_simulation(self):
        if len(self.PARAMS) is None:
            raise Exception('Please generate params and apply them before running simulation. apply_params()')


        positions = []
        velocities = []

        lorenz = self.__generate_lorenz(self.PARAMS['TIME_STEPS'], self.PARAMS['L_SAMPLING_RATE'], self.PARAMS['X_LORENZ'], self.PARAMS['Y_LORENZ'], self.PARAMS['Z_LORENZ'])

        #nts repeatedly dry running the oop version to find and work through bugs, currently trying to fix spawnbounds which was originally 'tupliised' in the get params in the original simulator fuckers
        spawn_bounds = (self.PARAMS['SPAWN_MIN'],self.PARAMS['SPAWN_MAX'])
        p, v = self.__generate_flock(self.PARAMS['BOID_COUNT'], spawn_bounds, self.PARAMS['RANDOM_VELOCITY'])
        positions.append(p)
        velocities.append(v)


        # below is a very excentric way of getting the number from the end of the thread name seen <Thread(ThreadPoolExecutor-0_0, started 6119583744)> (which is what current_thread() returns in a MT scenario)
        # otherwise rely on the failure to make the letter d an int to state that its a single threading scenario and thread indent should be 0 LOL
        # not too worried about the bad practise here considering the code is purely for aesthetics
        try: 
            thread_indent = int(threading.current_thread().name[-1])+1
        except:
            thread_indent=0

        for t in trange(self.PARAMS['TIME_STEPS'] - 1,desc=f"Thread: {threading.current_thread().name}",position=thread_indent,leave=False):
            new_v, new_x = self.__evolve(t,positions, velocities, lorenz[t])
            positions.append(new_x)
            velocities.append(new_v)

        return {
            "positions": positions,
            "velocities": velocities,
            "predator_positions": lorenz,
            "time_steps": self.PARAMS['TIME_STEPS'],
            "boid_count": self.PARAMS['BOID_COUNT'],
            "bounds": spawn_bounds,
        }

#----------------------------------------------------------------

#----------------------------------------------------------------
def main(custom_args=None):
    os.makedirs(DEFAULT_SAVE_PATH, exist_ok=True)
    #below accounts for main being called by other scripts, where custom args is a custom object
    if parser is None:
        try:
            args = custom_args
            args.view
        except:
            raise Exception("Invalid custom args. Must be 'object-like' and have accessable paramaters. Unused parameters must be set to None")
    else:
        args = parser.parse_args()

    Simulator = BoidSimulator()
    ParamManager, SaverLoader = Simulator.initialise()
    # if generating params, early return because this is an iscolated use case
    if args.generate_params:
        out_path = f"{args.generate_params}default_params.ini"
        print(f'Generating new parameters file at {out_path}')
        ParamManager.write_default_ini(out_path)
        return


    view = args.view # view some number of previous runs (.nz) 
    run = args.run # do a run returning the result
    iterations = args.iterations # Called replicas because they have different start conditions
    seed = args.seed
    save_path = args.save # save a run (given run or runview)
    multithread = args.multithread
    
    we_be_saving = save_path is not None

    # Checks
    print(f"run: {run}\niterations {iterations}\nseed: {seed}\nsave: {save_path}\nmultithreading: {multithread}")

    if not we_be_saving: 
        input('########## WARNING ##########\nNo save path, run will not be saved (dry run) abort CTRL-C or any key to continue with dry run')


    #(1)
    if view and not run:
        datas = SaverLoader.find_npzs(view)
        SimViewer(datas)
        return 

    #(2)
    if run:
        ini_path = run
        ParamManager.load_params_from_ini(ini_path)
        if we_be_saving: SaverLoader.set_save_dir(save_path)

        # setting the seed
        Simulator.set_seed(seed)


        if multithread:
            datas = [None] * iterations
            executor = TPE(max_workers=4)
            futures = {executor.submit(Simulator.run_simulation):i for i in range(iterations)}
            for f in tqdm(as_completed(futures),total=len(futures),desc='Iterations',position=0):
                i = futures[f]
                result = f.result()
                datas[i] = result
                
                if we_be_saving: SaverLoader.save_run(result,config_title=run)
                executor.shutdown()
        else:
            datas=[]
            for i in trange(iterations,desc='Iterations',position=0):
                result = Simulator.run_simulation()
                datas.append(result) 
                if we_be_saving: SaverLoader.save_run(result,config_title=run)



        if view: SimViewer(datas)        
        
        return datas

    raise Exception('No arguments were given but parser help cant be printed')


if __name__ == "__main__":
    main()