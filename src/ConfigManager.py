import configparser
import os

GLOBAL_DEFAULTS = {
        # Simulation params
        "sim":
            {
                "delta_t": 0.02,
                "simulation_steps": 500,
                "boid_count": 200,
                "spawn_min": -1.0,
                "spawn_max": 1.0,
                "sim_width": 10,
                "predator":True,
                "coord_system":'flat',#can be flat or torus
                "random_velocity": False,#whether or not to overide the seed when calculating the positions
                "random_position": False,#whether or not to overide the seed when calculating the velocities
                "random_seed":1,
            },

        # force constants
        "forces":
            {
                "k_speed" : 10.0,
                "k_repulsion" : 1.0,
                "k_alignment" : 0.1,
                "k_homing" : 2.0,
                "k_friction" : 5.0,
                "k_predator"  : 100.0
            },

        # sigmoidal function
        "sigmoid":
            {
                "beta" : 0.1,
                "alpha" : 200.0
            },

        # neighbour radii
        "radii":
            {
                "rad_alignment" : 1.0,
                "rad_repulsion" : 1.0,
                "rad_homing" : 1.0, #only relevent if torus coord space
                "rad_predator" : 2.0
            },

        # Lorenz conditions
        "lorenz":
            {
                "l_sigma" : 10.0,
                "l_rho" : 28.0,
                "l_beta" : 8/3,
                "x_lorenz" : 0.0,
                "y_lorenz" : 1.0,
                "z_lorenz" : 1.05,
                "l_sampling_rate" : 0.02 # number of sample per time step, also kind of predator speed
            }
    }

def load_config(path):
    """
    Load a `.ini` config file as a dictionary.

    Parameters
    ----------
    path : `str`
        Path to the `.ini` config file will be loaded.

    Returns
    -------
    dict
        parameter values flattened into one dict without .ini headings.

    Raises
    ------
    TypeError
        _If `name` has an invalid file extension._
    FileNotFoundError
        _If `path` does not exist._
    """    

    if isinstance(path,bool) or not path.endswith('.ini'):
        raise TypeError('Expected path to .ini file. e.g. PATH/TO/INI.ini')
    elif not os.path.exists(path):
        raise FileNotFoundError("Path does not exist.")

    cfg = configparser.ConfigParser()
    cfg.read(path)
    
    params = {}
    for section_title,section in GLOBAL_DEFAULTS.items():
        for k,v in section.items():
            cast = type(v)
            lower_key = k
            if cfg.has_option(section_title,lower_key):
                if cast == bool:
                    params[k] = cfg.getboolean(section_title,lower_key)
                else:
                    params[k] = cast(cfg.get(section_title,lower_key))
            else:
                raise NameError(f'\nConfig {path} is missing a definition for [{section_title}] {k} \n\t Please update the .ini to use this config file!')
    
    return params


def generate_config(path,name='config.ini',**kwargs):
    """
    Used to generate a `.ini` config from which simulator runs can be run

    Parameters
    ----------
    path : str
        Directory path in which the `.ini` config file will be saved.
    name : str, optional
        Name given to the `.ini` file, by default 'config.ini'

    Raises
    ------
    TypeError
        _If `name` has an invalid file extension._
    """    
    
    split = name.split('.')
    if len(split) == 2:
        if name.endswith('.ini'):
            name_val = name
        else:
            raise TypeError(f"{name} is not a valid extension for the config file. Must be a .ini or dont specify and .ini will be added automatically.")
    elif len(split) > 2:
        raise TypeError(f"{name} - invalid extension. Probably too many dots.")
    else:
        name_val = name + '.ini'
    
    ini_dir = os.path.join(path, name_val)

    cfg = configparser.ConfigParser()
    for section_title,section in GLOBAL_DEFAULTS.items():
        cfg[section_title] = {}
        for k,v in section.items():
            cfg[section_title][k] = str(v)
    
    
    with open(ini_dir,'w',encoding="utf-8") as f:
        cfg.write(f)


def compare_params(params:list, to_check=None):
    """
    Pass two dictionaries containing simulation parameters to compare whether or not they are the same.

    Parameters
    ----------
    params : list(dict)
        List config dictionaries
    to_check : str|list (Optional)
        single string or list of strings of parameters to compare against. Defaults to none, which checks against every parameter

    Returns
    -------
    `tuple[bool,str]`
    
    bool
        True or False, whether or not the given parameters are the same.
    str
        Table comparing their differences, can be appended to any string and or printed out for debugging purposes.
    """
    if len(params) == 1:
        return True,"One data provided, no comparison."
    if isinstance(to_check, str): 
        to_check = [to_check]
    elif to_check is None:
        to_check = set()
        for p in params:
            to_check.update(list(p.keys()))
        #technical the answer is here, comparing set length against each p.keys length, but an early exit would stop the string print logic

    same = True
    strout = "\n"
    rows = []

    for k in to_check:
        vals = [p.get(k, "MISSING") for p in params]
        is_consistent = all(v == vals[0] for v in vals) and "MISSING" not in vals
        status = "same" if is_consistent else "different"
        if not is_consistent: same = False
        rows.append((k, vals, status))

    if not same:
        col_width = 18
        headers = "".join([f"param_{i:<{col_width-6}}" for i in range(len(params))])
        strout += f"\n{'key':<20} {headers} status\n"
        strout += "-" * (20 + (col_width * len(params)) + 10)
        for k, vals, status in rows:
            val_line = "".join([f"{str(v):<{col_width}}" for v in vals])
            strout += f"\n{str(k):<20} {val_line}{status}"

    return same,strout