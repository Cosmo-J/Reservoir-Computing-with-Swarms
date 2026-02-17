from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from types import SimpleNamespace as sn
from tqdm import trange
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import configparser
from viewer import View as SimViewer
from scipy.spatial import cKDTree
import threading

from concurrent.futures import ThreadPoolExecutor as TPE, as_completed



parser = argparse.ArgumentParser()
parser.add_argument('--run',            '-r',   type=str,   nargs='?', const=True,  help="Run a new simulation given a .ini file path of parameters.")
parser.add_argument('--run-view',     '-rv',    type=str,   nargs='?', const=True,  help="Like -r but automatically begins viewing the simulation(s).")
parser.add_argument('--iterations',     '-i',   type=int,   nargs='?', default=1,   help="Use with run or runview to perform multiple simulations.")
parser.add_argument('--seed',                   type=int,   nargs='?', const=1,     help="Give a random seed to produce identical runs. Default seed is '1'.")
parser.add_argument('--generate-params','-g',   type=str,   nargs='?', const='.',   help="generate an empty .ini with default parameters.")
parser.add_argument('--view',           '-v',   type=str,   nargs='+',              help="View a previous simulation, given the file path of a valid '.npz'.")
parser.add_argument('--save',           '-s',   type=str,   nargs='?', const=False,  help="If iterations is more than 1, this parameter is taken as the save directory. Otherwise")
parser.add_argument('--multithread',    '-mt',   type=bool, nargs='?', default=False,  const=True, help="set true ")

# Optimisation focused definition
EMPTY_0x2 = np.empty((0, 2))
EPS =1e-12 # used for avoiding divide by 0
DEFAULT_SAVE_PATH = 'tests/boid_runs'

# these ones are how lymburn inits it's variables
DEFAULTS = {
# Simulation params
"DELTA_T": 0.02,
"TIME_STEPS": 100,
"BOID_COUNT": 200,
"SPAWN_MIN": -1.0,
"SPAWN_MAX": 1.0,
"RANDOM_VELOCITY": False,

# force constants
"K_SPEED" : 10.0,
"K_REPULSION" : 1,
"K_ALIGNMENT" : 0.1,
"K_HOMING" : 2.0,
"K_FRICTION" : 20.0,
"K_PREDATOR"  : 100,

# sigmoidal function
"BETA" : 0.1,
"ALPHA" : 200.0,

# neighbour radii
"RAD_ALIGNMENT" : 1,
"RAD_REPULSION" : 1,
"RAD_PREDATOR" : 1,

# Lorenz conditions
"L_SIGMA" : 10.0,
"L_RHO" : 28.0,
"L_BETA" : 8/3  ,
# Initial Positions
"X_LORENZ" : 0.0,
"Y_LORENZ" : 1.0,
"Z_LORENZ" : 1.05,

"L_TIME_STEPS" : 100,
"L_SAMPLING_RATE" : 0.02 # number of sample per time step, also kind of predator speed
}

# Boid Functions----------------------------------------------------------------

## Neighbour Functions
def update_neighbours(neighbour_dict,boid_positions):
    """
    key:\n
    0 - no neighbours\n
    1 - alignment and repulsion\n
    2 - alignemnt\n
    3 - repulsion
    """
    for boid_index,v in neighbour_dict.items():
        for sub_boid_index,_ in enumerate(v):
            if boid_index == sub_boid_index:#ignore self
                neighbour_dict[boid_index][sub_boid_index] = 0
                continue

            boid_pos = boid_positions[boid_index]
            sub_boid_pos = boid_positions[sub_boid_index]

            new_neighbour_value = 0 

            d = np.linalg.norm(boid_pos - sub_boid_pos)

            if d <= RAD_REPULSION and d <= RAD_ALIGNMENT:
                new_neighbour_value = 1

            elif d > RAD_REPULSION and d <= RAD_ALIGNMENT:
                new_neighbour_value = 2

            elif d <= RAD_REPULSION and d > RAD_ALIGNMENT:
                new_neighbour_value = 3

            neighbour_dict[boid_index][sub_boid_index] = new_neighbour_value

def get_neighbours(n,positions,velocities):
    a = []
    r = []

    for i, neighbour_type in enumerate(n): #the index of a neighbour type corresponds with the index of a given boid
        if neighbour_type == 1: # repulsion and alignment
            a.append(i)
            r.append(i)
        elif neighbour_type == 2: # just alignment
            a.append(i)
        elif neighbour_type == 3: # just repulsion
            r.append(i)
    
    a_x = [positions[index] for index in a] # the subset of boid positions who are attraction neighbours
    a_v = [velocities[index] for index in a]# the subset of boid velocities who are attraction neighbours
    a = sn(positions=a_x,velocities=a_v) # object with (positions,velocities) for the subset of boids who are attraction neighbours
    
    r_x = [positions[i] for i in r]
    r_v = [velocities[i] for i in r]
    r = sn(positions=r_x,velocities=r_v)

    return a, r

## Forces
def repulsion_force(boid,neis_x):
    """ 
        boid is an np.array(2) [x,y] of a given boid
        neighoburs is an np.array(2,n) where n is the number of neighbours
    """
    if len(neis_x)==0: return np.zeros(2)

    d = boid - neis_x
    denom = (d[:,0]**2 + d[:,1]**2) + EPS
    return (d / denom[:,None]).sum(axis=0)

    force = np.zeros(2)
    for n in neis_x:
        if np.array_equal(n,boid): continue  # boids consider themselves neighbours
        
        numerator = boid - n
        denom = np.linalg.norm(boid-n) ** 2
        force += (numerator/denom)
    return force

def alignment_force(boid_v,neis_v):
    """boid is an np.array(2) [x,y] of a given boid's velocity\n neighoburs is an np.array(2,n) where n is the number of neighbours, and gives the velocities of all the neighbours"""
    force = np.array([0.0,0.0])
    for n in neis_v:
        force+= n - boid_v
    return force

def homing_force(boid,home=np.array([0.0,0.0])):
    return home-boid

def friction_force(boid_v):
    speed = np.hypot(boid_v[0],boid_v[1])
    return -boid_v * ((speed - K_SPEED) / K_SPEED)

    denom = (np.linalg.norm(boid_v) - K_SPEED) * (boid_v*-1)
    force = denom/K_SPEED
    return force

def predator_force(boid_x,pred_x):
    if pred_x is None: return np.array([0.0,0.0])
    d= np.linalg.norm(boid_x - pred_x)

    # this 'if else' is the heaviside function
    if(d<=RAD_PREDATOR):
        numer = boid_x-pred_x
        denom = d**2
        return (numer/denom)
    else:
        return np.array([0.0,0.0])

def total_force(boid_x,boid_v,a_neighbours,r_neighbours,pred_x=None):
#            |-coefficent---|-force---------|-force-params---------|
    force = ((K_ALIGNMENT*  alignment_force (boid_v,a_neighbours)) +
            (K_REPULSION *  repulsion_force (boid_x,r_neighbours)) +
            (K_FRICTION  *  friction_force  (boid_v)) +
            (K_HOMING    *  homing_force    (boid_x)) +
            (K_PREDATOR  *  predator_force  (boid_x,pred_x) ))       

    # sigmoidal function
    force_sigmoid = ALPHA * np.tanh(BETA * force)

    return force_sigmoid

def force_matrix(boid_xs,boid_vs,pred_x=None):
    forces = np.empty((len(boid_xs),2))

    tree = cKDTree(boid_xs)
    align_lists = tree.query_ball_point(boid_xs,r=RAD_ALIGNMENT)
    repulsion_lists = tree.query_ball_point(boid_xs,r=RAD_REPULSION)



    for i, (x,v) in enumerate(zip(boid_xs,boid_vs)):
        a_idx = [j for j in align_lists[i] if j != i]
        r_idx = [j for j in repulsion_lists[i] if j != i]

        a_neis_v = boid_vs[a_idx] if a_idx else EMPTY_0x2
        r_neis_x = boid_xs[r_idx] if r_idx else EMPTY_0x2

        forces[i] = total_force(x,v,a_neis_v,r_neis_x,pred_x)


    return forces

def lorenz_equations(t,start_states):
    x,y,z = start_states
    dxBYdt = L_SIGMA * (y-x)
    dyBYdt = x * (L_RHO - z) - y
    dzBYdt = (x * y) - (L_BETA * z)
    return dxBYdt,dyBYdt,dzBYdt

def rescale(axis):
    return 2 * (axis- np.mean(axis)) / np.std(axis)

def generate_lorenz(time_steps, sample_rate, x_init, y_init, z_init):
    lorenz_segment = time_steps*sample_rate

    soln = solve_ivp(lorenz_equations, t_span=(0,lorenz_segment) ,y0=(x_init,y_init,z_init) ,dense_output=True)
    t = np.linspace(0, lorenz_segment, time_steps)
    coords = soln.sol(t).T

    rescaled_x_coords = rescale(coords[:, 0])
    rescaled_y_coords = rescale(coords[:, 1])
    return np.column_stack((rescaled_x_coords,rescaled_y_coords))

def generate_flock(flock_size,lim,random_velocity=False):
    #x = np.random.uniform(lim[0], lim[1], size=(flock_size, 2))
    x = lim[0] + (lim[1] - lim[0]) * np.random.beta(2, 2, size=(flock_size, 2))

    if random_velocity: 
        v=np.random.rand(flock_size,2)
    else: 
        v = np.zeros((flock_size,2),dtype=float)

    # returns positions, velocities, neighbour data
    return x, v

def evolve(t,positions,velocities,prior_lorenz_x):
    '''
    the priors are the given parameter at t
    '''
    current_x = positions[t]
    current_v = velocities[t]

    #2. Create force matrix
    fm = force_matrix(current_x,current_v,prior_lorenz_x)

    #3. update velocity matrix
    #4. update position matrix
    new_v = current_v + (fm * DELTA_T)
    new_x = current_x + (new_v * DELTA_T)

    return new_v,new_x

def run_simulation():
    positions = []
    velocities = []

    lorenz = generate_lorenz(TIME_STEPS, L_SAMPLING_RATE, X_LORENZ, Y_LORENZ, Z_LORENZ)

    p, v = generate_flock(BOID_COUNT, SPAWN_BOUNDS, RANDOM_VELOCITY)
    positions.append(p)
    velocities.append(v)


    # below is a very excentric way of getting the number from the end of the thread name seen <Thread(ThreadPoolExecutor-0_0, started 6119583744)> (which is what current_thread() returns in a MT scenario)
    # otherwise rely on the failure to make the letter d an int to state that its a single threading scenario and thread indent should be 0 LOL
    # not too worried about the bad practise here considering the code is purely for aesthetics
    try: 
        thread_indent = int(threading.current_thread().name[-1])+1
    except:
        thread_indent=0

    for t in trange(TIME_STEPS - 1,desc=f"Thread: {threading.current_thread().name}",position=thread_indent,leave=False):
        new_v, new_x = evolve(t,positions, velocities, lorenz[t])
        positions.append(new_x)
        velocities.append(new_v)

    return {
        "positions": positions,
        "velocities": velocities,
        "predator_positions": lorenz,
        "time_steps": TIME_STEPS,
        "boid_count": BOID_COUNT,
        "bounds": SPAWN_BOUNDS,
    }


#----------------------------------------------------------------
# some functions below were initially written by CHATGPT, but reviewed and modified by myself 
def load_ini(path: str) -> dict:
    cfg = configparser.ConfigParser()
    cfg.read(path)

    # Expect sections; fall back to DEFAULTS if missing.
    def get(section, key, cast):
        if cfg.has_option(section, key):
            return cast(cfg.get(section, key))
        
        #return DEFAULTS[key]
        else:
            raise Exception(f'Config {path} is missing a key:value for {section} {key}\n\tPlease update the .ini to use this config file')
        

    params = {}

    params["K_SPEED"] = get("forces", "K_SPEED", float)
    params["K_REPULSION"] = get("forces", "K_REPULSION", float)
    params["K_ALIGNMENT"] = get("forces", "K_ALIGNMENT", float)
    params["K_HOMING"] = get("forces", "K_HOMING", float)
    params["K_FRICTION"] = get("forces", "K_FRICTION", float)
    params["K_PREDATOR"] = get("forces", "K_PREDATOR", float)

    params["ALPHA"] = get("sigmoid", "ALPHA", float)
    params["BETA"] = get("sigmoid", "BETA", float)

    params["RAD_REPULSION"] = get("radii", "RAD_REPULSION", float)
    params["RAD_ALIGNMENT"] = get("radii", "RAD_ALIGNMENT", float)
    params["RAD_PREDATOR"] = get("radii", "RAD_PREDATOR", float)

    params["DELTA_T"] = get("sim", "DELTA_T", float)
    params["TIME_STEPS"] = get("sim", "TIME_STEPS", int)
    params["BOID_COUNT"] = get("sim", "BOID_COUNT", int)
    params["SPAWN_MIN"] = get("sim", "SPAWN_MIN", float)
    params["SPAWN_MAX"] = get("sim", "SPAWN_MAX", float)
    params["RANDOM_VELOCITY"] = get("sim", "RANDOM_VELOCITY", lambda s: str(s).lower() in ("1", "true", "yes", "y"))

    params["L_SIGMA"] = get("lorenz", "L_SIGMA", float)
    params["L_RHO"] = get("lorenz", "L_RHO", float)
    params["L_BETA"] = get("lorenz", "L_BETA", float)
    params["X_LORENZ"] = get("lorenz", "X_LORENZ", float)
    params["Y_LORENZ"] = get("lorenz", "Y_LORENZ", float)
    params["Z_LORENZ"] = get("lorenz", "Z_LORENZ", float)
    params["L_SAMPLING_RATE"] = get("lorenz", "L_SAMPLING_RATE", float)

    return params

def write_default_ini(path: str):
    cfg = configparser.ConfigParser()
    cfg["sim"] = {
        "DELTA_T": str(DEFAULTS["DELTA_T"]),
        "TIME_STEPS": str(DEFAULTS["TIME_STEPS"]),
        "BOID_COUNT": str(DEFAULTS["BOID_COUNT"]),
        "SPAWN_MIN": str(DEFAULTS["SPAWN_MIN"]),
        "SPAWN_MAX": str(DEFAULTS["SPAWN_MAX"]),
        "RANDOM_VELOCITY": str(DEFAULTS["RANDOM_VELOCITY"]),
    }
    cfg["forces"] = {
        "K_SPEED": str(DEFAULTS["K_SPEED"]),
        "K_REPULSION": str(DEFAULTS["K_REPULSION"]),
        "K_ALIGNMENT": str(DEFAULTS["K_ALIGNMENT"]),
        "K_HOMING": str(DEFAULTS["K_HOMING"]),
        "K_FRICTION": str(DEFAULTS["K_FRICTION"]),
        "K_PREDATOR": str(DEFAULTS["K_PREDATOR"]),
    }
    cfg["sigmoid"] = {
        "ALPHA": str(DEFAULTS["ALPHA"]),
        "BETA": str(DEFAULTS["BETA"]),
    }
    cfg["radii"] = {
        "RAD_REPULSION": str(DEFAULTS["RAD_REPULSION"]),
        "RAD_ALIGNMENT": str(DEFAULTS["RAD_ALIGNMENT"]),
        "RAD_PREDATOR": str(DEFAULTS["RAD_PREDATOR"]),
    }
    cfg["lorenz"] = {
        "L_SIGMA": str(DEFAULTS["L_SIGMA"]),
        "L_RHO": str(DEFAULTS["L_RHO"]),
        "L_BETA": str(DEFAULTS["L_BETA"]),
        "X_LORENZ": str(DEFAULTS["X_LORENZ"]),
        "Y_LORENZ": str(DEFAULTS["Y_LORENZ"]),
        "Z_LORENZ": str(DEFAULTS["Z_LORENZ"]),
        "L_SAMPLING_RATE": str(DEFAULTS["L_SAMPLING_RATE"]),
    }

    with open(path, "w", encoding="utf-8") as f:
        cfg.write(f)

def apply_params(params: dict = DEFAULTS):
    global K_SPEED, K_REPULSION, K_ALIGNMENT, K_HOMING, K_FRICTION, K_PREDATOR
    global ALPHA, BETA
    global RAD_REPULSION, RAD_ALIGNMENT, RAD_PREDATOR
    global L_SIGMA, L_RHO, L_BETA, X_LORENZ, Y_LORENZ, Z_LORENZ, L_SAMPLING_RATE
    global TIME_STEPS, BOID_COUNT, SPAWN_BOUNDS, RANDOM_VELOCITY, DELTA_T
    
    DELTA_T = float(params["DELTA_T"])
    TIME_STEPS = int(params["TIME_STEPS"])
    BOID_COUNT = int(params["BOID_COUNT"])
    SPAWN_BOUNDS = [float(params["SPAWN_MIN"]), float(params["SPAWN_MAX"])]
    RANDOM_VELOCITY = bool(params["RANDOM_VELOCITY"])
    
    K_SPEED = float(params["K_SPEED"])
    K_REPULSION = float(params["K_REPULSION"])
    K_ALIGNMENT = float(params["K_ALIGNMENT"])
    K_HOMING = float(params["K_HOMING"])
    K_FRICTION = float(params["K_FRICTION"])
    K_PREDATOR = float(params["K_PREDATOR"])

    ALPHA = float(params["ALPHA"])
    BETA = float(params["BETA"])

    RAD_REPULSION = float(params["RAD_REPULSION"])
    RAD_ALIGNMENT = float(params["RAD_ALIGNMENT"])
    RAD_PREDATOR = float(params["RAD_PREDATOR"])


    L_SIGMA = float(params["L_SIGMA"])
    L_RHO = float(params["L_RHO"])
    L_BETA = float(params["L_BETA"])
    X_LORENZ = float(params["X_LORENZ"])
    Y_LORENZ = float(params["Y_LORENZ"])
    Z_LORENZ = float(params["Z_LORENZ"])
    L_SAMPLING_RATE = float(params["L_SAMPLING_RATE"])

def save_run(data: dict, name: str | None = None, out_dir: str = "boid_runs",params:dict = None,config_title:str = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if not name:
        name = datetime.now().strftime("%d-%m-%Y-%H%M-%S")
    
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez(
        path,
        positions=data["positions"],
        velocities=data["velocities"],
        predator_positions=data["predator_positions"],
        time_steps=data["time_steps"],
        boid_count=data["boid_count"],
        bounds=data["bounds"],
        config=params,
        config_title = [config_title]
    )
    return path

def find_npzs(paths:str):
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
    datas =[load_run(f) for f in npzs]
    print(f'Loaded {len(datas)} run(s)')

    return datas

def load_run(path: str) -> dict:
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

def find_dir(path:str,name:str = False,its=0):
    '''
    Docstring for find_or_make_dir
    :return: tuple(save path, save name) savename will default to datetime if no option provided
    '''
    if its > 1: raise Exception(f'Could not find path {path}')

    time_now = datetime.now().strftime('%d-%m-%Y-%H%M-%S')

    if os.path.exists(path):
        if os.path.isdir(path):
            if name: return name, path
            else:    return time_now, path
        else:
            print(f'{path} already exists as a file')
            raise FileExistsError
    else:
        dirname = os.path.dirname(path)
        name = os.path.basename(path)
        return find_dir(dirname,name,1)
        

#----------------------------------------------------------------

def TestSimulation(config_path):
    # find ini file
    # apply params
    # return the associated arrays positions instead of saving
    params = load_ini(config_path)
    return run_simulation(params)

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

    # if generating params, early return because this is an iscolated use case
    if args.generate_params:
        print(f'Generating new parameters file at {out_path}')
        out_path = "default_params.ini"
        write_default_ini(out_path)
        return


    view = args.view # view some number of previous runs (.nz) 
    run = args.run # do a run returning the result
    runview = args.run_view # do a run returning the result and displaying 
    iterations = args.iterations # Called replicas because they have different start conditions
    seed = args.seed
    save = args.save # save a run (given run or runview)
    multithread = args.multithread

    print(f"run: {run}\nrunview: {runview}\niterations {iterations}\nseed: {seed}\nsave: {save}")

    if (save is None):
        save = input('########## WARNING ##########\nNo save path, run will not be saved (dry run) abort CTRL-C or input savepath now:')


    # setting the seed
    if isinstance(seed,int): 
        np.random.seed(seed)
        print(f"Using custom np.random seed: {seed}")
    else: 
        np.random.seed(1)


    if view:
        datas = find_npzs(view)
        SimViewer(datas)
        return
    
    elif run or runview:
        # loading parameters
        if run: ini_path = run
        elif runview: ini_path = runview


        if not isinstance(ini_path,str) or ini_path is None:
            print("---USING DEFAULT PARAMETERS---")
            params = DEFAULTS
        else:
            params = load_ini(ini_path)
        
        apply_params(params)

        if save is not None:
            if not save: 
                save_name, save_path = find_dir(DEFAULT_SAVE_PATH)
            else:
                save_name, save_path = find_dir(save)

            max_number = 0
            name_it = save_name

            #TODO known bug occurs if e.g. file10 file13, code will assume file10 is the max
            while os.path.exists(f'{save_path}/{name_it+str(max_number)}.npz'):
                max_number+=1
                
            print(f'MAX IT FOUND WITH NAME: {name_it+str(max_number-1)}')

    

        if multithread:
            datas = [None] * iterations
            executor = TPE(max_workers=4)
            futures = {executor.submit(run_simulation):i for i in range(iterations)}
            for f in tqdm(as_completed(futures),total=len(futures),desc='Iterations',position=0):
                i = futures[f]

                result = f.result()
                datas[i] = result

                if save is not None:
                    save_name_iterated = save_name + f'{max_number+i}'
                    save_run(result,name=save_name_iterated,out_dir=save_path,params=params,config_title=ini_path)  

            executor.shutdown()
        else: 
            datas=[]
            for i in trange(iterations,desc='Iterations',position=0):
                datas.append(run_simulation())

                if save is not None:
                    run_num = 0
                    save_name_iterated=save_name+f"{run_num}"

                    # make sure to give unique filename
                    while os.path.exists(f'{save_path}/{save_name_iterated}.npz'): 
                        run_num+=1
                        save_name_iterated=save_name+f"{run_num}"
                    saved_path = save_run(datas[i],name=save_name_iterated, out_dir=save_path,params=params,config_title=save_name)
                
                    print(f"Saved run(s) to: {saved_path}")

        if runview:
            SimViewer(datas)        
        
        return datas

    raise Exception('No arguments were given but parser help cant be printed')

if __name__ == "__main__":
    main()