from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from types import SimpleNamespace as sn
from tqdm import trange
from copy import deepcopy
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import configparser
from viewer import SimViewer

parser = argparse.ArgumentParser()

parser.add_argument('--run','-r', help="Run a new simulation given a .ini file path of parameters.", nargs='?', const=True)
parser.add_argument('--runview','-rv', help="Like -r but automatically begins viewing the", nargs='?', const=True)
parser.add_argument('--genps','-g',type=str, help="generate an empty .ini with default parameters", nargs='?', const='.')
parser.add_argument('--view','-v',type=str, help="View a previous simulation, given the file path of a valid .npz")
parser.add_argument('--save','-s', help="Save the run as an .npz, given a string name (will default to date time)")

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

# time step / simulation
"DELTA_T" : 0.02,

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

# Boid Functions

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
def repulsion_force(boid,neis_x,debug=False):
    """ 
        boid is an np.array(2) [x,y] of a given boid
        neighoburs is an np.array(2,n) where n is the number of neighbours
    """
    force = np.array([0.0,0.0])
    for n in neis_x:
        if np.array_equal(n,boid): continue  # boids consider themselves neighbours
                                # this is useful for finding the avg position in homing, but should be skipped here

        numerator = boid - n

        denom = np.linalg.norm(boid-n) ** 2
        force += (numerator/denom)

        if(debug): 
            print(f'Force total: {force}')
            print(f'\t{numerator} / {denom} = {numerator/denom}')
            print(f'\tXi = {boid}, Xj = {n}')
    
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

def total_force(boid_x,boid_v,a_neighbours,r_neighbours,pred_x=None,debug=False):
#            |-coefficent---|-force---------|-force-params---------|
    force = ((K_ALIGNMENT*  alignment_force (boid_v,a_neighbours.velocities)) +
            (K_REPULSION *  repulsion_force (boid_x,r_neighbours.positions)) +
            (K_FRICTION  *  friction_force  (boid_v)) +
            (K_HOMING    *  homing_force    (boid_x)) +
            (K_PREDATOR  *  predator_force  (boid_x,pred_x) ))       

    # sigmoidal function
    force_sigmoid = ALPHA * np.tanh(BETA * force)

    # debug showing all the forces
    if(debug):
        print(   
                f'Force: {force} Force Sigmoid: {force_sigmoid}',
                f'\n\tFa: { K_ALIGNMENT*alignment_force(boid_v,a_neighbours.velocities) }',
                f'\n\tFr: { K_REPULSION*repulsion_force(boid_x,r_neighbours.positions) }',
                f'\n\tFf: { K_FRICTION*friction_force(boid_v) }',
                f'\n\tFh: { K_HOMING*homing_force(boid_x) }',
                f'\n\tFp: { K_PREDATOR *predator_force(boid_x,pred_x) }')
    
    return force_sigmoid

def force_matrix(boid_xs,boid_vs,ns,pred_x=None,debug=False):
    forces = np.empty((len(boid_xs),2))
    for i, (x,v,n) in enumerate(zip(boid_xs,boid_vs,ns.values())):

        attraction_neighbours, repulsion_neighbours = get_neighbours(n,boid_xs,boid_vs)

        forces[i] = total_force(x,v,attraction_neighbours,repulsion_neighbours,pred_x,debug)
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
    soln = solve_ivp(lorenz_equations, t_span=(0,time_steps) ,y0=(x_init,y_init,z_init) ,dense_output=True)
    t = np.linspace(0, time_steps, int(time_steps/sample_rate))
    coords = soln.sol(t).T

    rescaled_x_coords = rescale(coords[:, 0])
    rescaled_y_coords = rescale(coords[:, 1])
    return np.column_stack((rescaled_x_coords,rescaled_y_coords))

def generate_flock(flock_size,lim,random_velocity=False):
    x = np.random.uniform(lim[0], lim[1], size=(flock_size, 2))
    if random_velocity: 
        v=np.random.rand(flock_size,2)
    else: 
        v = np.zeros((flock_size,2),dtype=float)

    # dictionary mapping the number tag of each boid to a bit array of its near neighbours
    n = {i:[0] * flock_size for i in range(flock_size)}

    # returns positions, velocities, neighbour data
    return x, v, n

def evolve(prior_x,prior_v,n,prior_lorenz_x):
    '''
    the priors are the given parameter at t
    '''
    new_x = deepcopy(prior_x)
    new_v = deepcopy(prior_v)
    
    

    #1. Calculate neighbours
        # given the neighbour dictionary (to be changed), and the positions of all the boids
    update_neighbours(n,prior_x)

    #2. Create force matrix
    fm = force_matrix(new_x,new_v,n,prior_lorenz_x,False)

    #3. update velocity matrix
    #4. update position matrix
    for i,v in enumerate(new_v):
        new_v[i]+=fm[i] * DELTA_T
        new_x[i]+= new_v[i] * DELTA_T

    return new_x,new_v,n


# functions below were initially written by CHATGPT, but reviewed and modified by myself 
def load_ini(path: str) -> dict:
    cfg = configparser.ConfigParser()
    cfg.read(path)

    # Expect sections; fall back to DEFAULTS if missing.
    def get(section, key, cast):
        if cfg.has_option(section, key):
            return cast(cfg.get(section, key))
        return DEFAULTS[key]

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

def run_simulation(params: dict):
    apply_params(params)

    positions = []
    velocities = []

    lorenz = generate_lorenz(TIME_STEPS, L_SAMPLING_RATE, X_LORENZ, Y_LORENZ, Z_LORENZ)

    p, v, n = generate_flock(BOID_COUNT, SPAWN_BOUNDS, RANDOM_VELOCITY)
    positions.append(p)
    velocities.append(v)
    neighbours = n

    for t in trange(TIME_STEPS - 1):
        new_p, new_v, neighbours = evolve(positions[t], velocities[t], neighbours, lorenz[t])
        positions.append(new_p)
        velocities.append(new_v)

    return {
        "positions": positions,
        "velocities": velocities,
        "predator_positions": lorenz,
        "time_steps": TIME_STEPS,
        "boid_count": BOID_COUNT,
        "bounds": SPAWN_BOUNDS,
    }

def save_run(data: dict, name: str | None = None, out_dir: str = "boid_runs") -> str:
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
    )
    return path

def load_run(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "positions": z["positions"],
        "velocities": z["velocities"],
        "predator_positions": z["predator_positions"],
        "time_steps": int(z["time_steps"]),
        "boid_count": int(z["boid_count"]),
        "bounds": z["bounds"],
    }


def main():
    args = parser.parse_args()
    print(args)

    # gen empty ini
    if args.genps:
        out_path = "default_params.ini"
        write_default_ini(out_path)
        print(f"Wrote default ini to: {out_path}")
        return

    # rerun / view an existing run
    if args.view:
        data = load_run(args.view)
        SimViewer(data)
        return

    # run a new simulation from ini
    if args.run or args.runview:
        ini_path = args.run if args.run else args.runview
        if not isinstance(ini_path,str) or ini_path is None:
            print("---USING DEFAULT PARAMETERS---")
        params = load_ini(ini_path)
        data = run_simulation(params)

        saved_path = None
        if args.save is not None:
            # If -s provided with a string, use it. If -s provided but empty (rare), default to datetime.
            save_name = args.save if isinstance(args.save, str) and args.save.strip() else None
            saved_path = save_run(data, save_name)
            print(f"Saved run to: {saved_path}")

        if args.runview:
            SimViewer(data)
        return

    # If no args: show help
    parser.print_help()









if __name__ == "__main__":
    main()