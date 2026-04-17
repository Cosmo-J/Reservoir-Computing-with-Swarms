import configparser
import os

GLOBAL_DEFAULTS = {
        # Simulation params
        "sim":
            {
                "DELTA_T": 0.02,
                "TIME_STEPS": 100,
                "BOID_COUNT": 200,
                "SPAWN_MIN": -1.0,
                "SPAWN_MAX": 1.0,
                "SIM_WIDTH": 10,
                "PREDATOR":True,
                "COORD_SYSTEM":'flat',#can be flat or torus
                "RANDOM_VELOCITY": False,#whether or not to overide the seed when calculating the positions
                "RANDOM_POSITION": False,#whether or not to overide the seed when calculating the velocities
                "RANDOM_SEED":1,
            },

        # force constants
        "forces":
            {
                "K_SPEED" : 10.0,
                "K_REPULSION" : 1.0,
                "K_ALIGNMENT" : 0.1,
                "K_HOMING" : 2.0,
                "K_FRICTION" : 20.0,
                "K_PREDATOR"  : 100.0
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
                "RAD_ALIGNMENT" : 1.0,
                "RAD_REPULSION" : 1.0,
                "RAD_HOMING" : 1.0,
                "RAD_PREDATOR" : 1.0
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

    if not path.endswith('.ini'):
        raise TypeError('Expected path to .ini file. e.g. PATH/TO/INI.ini')
    elif not os.path.exists(path):
        raise FileNotFoundError("Path does not exist.")

    cfg = configparser.ConfigParser()
    cfg.read(path)
    
    params = {}
    for section_title,section in GLOBAL_DEFAULTS.items():
        for k,v in section.items():
            cast = type(v)
            lower_key = k.lower()
            if cfg.has_option(section_title,lower_key):
                if cast == bool:
                    params[k] = cfg.getboolean(section_title,lower_key)
                else:
                    params[k] = cast(cfg.get(section_title,lower_key))
            else:
                raise NameError(f'Config {path} is missing a key:value for {section_title} {k} \n\t Please update the .ini to use this config file!')
    
    return params


def generate_config(path,name='config.ini'):
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


def compare_params(param1, param2):
    """
    Pass two dictionaries containing simulation parameters to compare whether or not they are the same.

    Parameters
    ----------
    param1 : dict
        Dictionary of parameters to bec compared with `param2`
    param2 : dict
        Dictionary of parameters to bec compared with `param1`

    Returns
    -------
    `tuple[bool,str]`
    
    bool
        True or False, whether or not the given parameters are the same.
    str
        Table comparing their differences, can be appended to any string and or printed out for debugging purposes.
    """
    same = True
    strout = "\n"
    
    rows = []
    for k in set(list(param1)+list(param2)):
        v1 = param1.get(k, "MISSING")
        v2 = param2.get(k, "MISSING")

        if k not in param2:
            same = False
            status = "missing in param2"
        elif k not in param1:
            same = False
            status = "missing in param1"

        if v1 == v2:
            status = "same"
        else:
            same = False
            status = "different"

        rows.append((k, v1, v2, status))

    if not same:
        strout+="\n"
        strout+= f"{'key':<20} {'param1':<20} {'param2':<20} status"
        strout+="\n"
        strout+= ("-" * 75)

        for k, v1, v2, status in rows:
            strout+="\n"
            strout+= f"{str(k):<20} {str(v1):<20} {str(v2):<20} {status}"
    
    return same,strout