import configparser
import os
class SimParams:
    DEFAULTS = {
            # Simulation params
            "sim":
                {
                    "DELTA_T": 0.02,
                    "TIME_STEPS": 100,
                    "BOID_COUNT": 200,
                    "SPAWN_MIN": -1.0,
                    "SPAWN_MAX": 1.0,
                    "RANDOM_VELOCITY": True,
                    "SIM_WIDTH": 10,
                    "PREDATOR":True,
                    'COORD_SYSTEM':'flat'#can be flat or torus
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

    def __init__(self):
        self.params = self.get_flat_params(self.DEFAULTS)

    
    def get_flat_params(self,params):
        '''
            use to flatten a dict which is storing its parameters two tiered like .ini format. For an example of this two teird thing, see the DEFUALT decleration in this class.
        '''
        flat = {}
        for section_title,section in params.items():
            for k,v in section.items():
                flat[k] = v
        return flat
    
    def load_params_from_ini(self,path):#also used to load from defaults
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
                lower_key = k.lower()
                if cfg.has_option(section_title,lower_key):
                    if cast == bool:
                        self.params[k] = cfg.getboolean(section_title,lower_key)
                    else:
                        self.params[k] = cast(cfg.get(section_title,lower_key))
                else:
                    input(f'Config {path} is missing a key:value for {section_title} {k} \n\t Please update the .ini to use this config file!\nType anything to continue with default: {v}')
                    self.params[k] = v
        return self.params
    
    def write_default_ini(self,path,name='config.ini'):
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
        for section_title,section in self.DEFAULTS.items():
            cfg[section_title] = {}
            for k,v in section.items():
                cfg[section_title][k] = str(v)
        
        
        with open(ini_dir,'w',encoding="utf-8") as f:
            cfg.write(f)
