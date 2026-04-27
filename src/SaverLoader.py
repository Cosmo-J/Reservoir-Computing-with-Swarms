from multiprocessing import Value
from os.path import isdir
import numpy as np
import os
import tempfile
import gc
import re

TMP_PATH = 'tmp'
PREDICTIONS_DIR_NAME = "predictions"
RUNS_DIR_NAME = "runs"
READOUTS_DIR_NAME = "readouts"
PCAS_DIR_NAME = "pcas"
SUB_DIR_NAMES = [PREDICTIONS_DIR_NAME, RUNS_DIR_NAME, READOUTS_DIR_NAME, PCAS_DIR_NAME]

class Saver:
    def __init__(self,save_path):
        self.tmp_path = TMP_PATH
        self.predictions_dir_name = PREDICTIONS_DIR_NAME
        self.runs_dir_name = RUNS_DIR_NAME
        self.readouts_dir_name = READOUTS_DIR_NAME
        self.pcas_dir_name = PCAS_DIR_NAME
        self.sub_dir_names = SUB_DIR_NAMES

        self.save_path = self._validate_save_path(save_path)

    
    def _unique_name(self,candidate_name:str):
        '''
        :candidate_name: this is the potential file name /path/to/file
        :return: Modified path /path/to/file_run_12.npz
        '''
        extension = '.npz'
        iter = ''
        candidate_path = os.path.join(self.save_path,f'{candidate_name}')

        while os.path.exists(candidate_path+str(iter)+extension):
            if isinstance(iter,str): 
                # this is for the first loop only
                iter = 0
            else: 
                iter+=1

        return candidate_path+str(iter)+extension

    def _validate_save_path(self, path:str):
        if os.path.isdir(path):
            print("Save dir found")
            for d in self.sub_dir_names:
                sub_dir = os.path.join(path,d)
                errors = []
                if not os.path.exists(sub_dir):
                    try:
                        os.makedirs(sub_dir)
                        print(f"\tcreated subpath {sub_dir}")
                    except Exception as e:
                        errors.append((e,f"{sub_dir}"))
                if len(errors)>1:
                    for e,failed_dir in errors:
                        print(f"\t{e} : {failed_dir}")
            return os.path.abspath(path)
        else:
            raise FileNotFoundError(f"Directory {path} doesn't exist or cant be found.")


    def save_run(self, data, prefix="", cleanup_tmps=False):
        assert os.path.exists(self.save_path), f"{self.save_path} - Save path no longer exists."
        if prefix=="": 
            name = f"{os.path.basename(self.save_path)}_run"
        else:
            if prefix.isalnum():
                name = f"{prefix}_run"
            else:
                raise ValueError(f"Suffix {prefix} is an invalid file name. Must be alpha numeric only.")
        main_save_path = self._unique_name(name) 
        name = os.path.join(self.runs_dir_name,name)
        subdir_save_path = self._unique_name(name) 

        candidate_paths = []
        candidate_paths.append(subdir_save_path)
        candidate_paths.append(main_save_path)

        errors = []
        for c_path in candidate_paths:
            try:
                np.savez(
                    c_path,
                    positions=data["positions"],
                    velocities=data["velocities"],
                    predator_positions=data["predator_positions"],
                    simulation_steps=data["simulation_steps"],
                    boid_count=data["boid_count"],
                    bounds=data["bounds"],
                    config=data['config'],
                )
                print(f"Saving - {c_path}")
                full_save_path_and_name = c_path
                break
            except:
                errors.append(c_path)


        if len(errors)>0:
            print("Saving incurred errors:")
            for e,path in errors:
                print(f"{e} : {path}")

        
        if isinstance(data["positions"],np.memmap) and cleanup_tmps:
            try:
                print("Attempting to clean up tempory files... ",end="")
                data["positions"].flush()
                data["velocities"].flush()

                tmp_pos_filename = data["positions"].filename
                tmp_vel_filename = data["velocities"].filename

                del data["positions"]
                del data["velocities"]

                gc.collect()
                error = 0
                if os.path.exists(tmp_vel_filename):
                    os.remove(tmp_vel_filename)
                else: error+=1
                if os.path.exists(tmp_pos_filename):
                    os.remove(tmp_pos_filename)
                else: error+=1
                
                if not error:
                    print("Success!")
                else:
                    print(f"\n({error} Errors) A tempory file was moved or already deleted. This is either good news or terrible news (joke).")
            except Exception as e:
                print(f"{e} : Temporary file cleanup failed due to unknown error.")

        return full_save_path_and_name

    def save_prediction(self,data,prefix=""):
        assert os.path.exists(self.save_path), f"{self.save_path} - Save path no longer exists."
        if prefix=="": 
            name = f"{os.path.basename(self.save_path)}_prediction"
        else:
            if prefix.isalnum():
                name = f"{prefix}_prediction"
            else:
                raise ValueError(f"Suffix {prefix} is an invalid file name. Must be alpha numeric only.")

        main_save_path = self._unique_name(name) 
        name = os.path.join(self.predictions_dir_name,name)
        subdir_save_path = self._unique_name(name) 

        candidate_paths = []
        candidate_paths.append(subdir_save_path)
        candidate_paths.append(main_save_path)

        errors = []
        for c_path in candidate_paths:
            try:
                np.savez(
                    c_path,
                    readout_method=data["readout_method"],
                    prediction_distance=data["prediction_distance"],
                    train_size=data["train_size"],
                    prediction=data["prediction"],
                    alpha=data["alpha"],
                    alpha_search=data["alpha_search"],
                    corr_coef=data["corr_coef"],
                    simulation_config=data['simulation_config'],
                    y_test=data['y_test'],
                )
                full_save_path_and_name = c_path
                break
            except InterruptedError:
                print("Couldn't save prediction!")
            
        if len(errors)>0:
            print("Saving incurred errors:")
            for e,path in errors:
                print(f"{e} : {path}")

        return full_save_path_and_name

    def save_readout(self,data,prefix="",cleanup_tmps=False):
        assert os.path.exists(self.save_path), f"{self.save_path} - Save path no longer exists."  
        if prefix=="": 
            name = f"{os.path.basename(self.save_path)}_readout"
        else:
            if prefix.isalnum():
                name = f"{prefix}_readout"
            else:
                raise ValueError(f"Suffix {prefix} is an invalid file name. Must be alpha numeric only.")
        main_save_path = self._unique_name(name) 
        name = os.path.join(self.readouts_dir_name,name)
        subdir_save_path = self._unique_name(name) 

        candidate_paths = []
        candidate_paths.append(subdir_save_path)
        candidate_paths.append(main_save_path)


        errors = []
        for c_path in candidate_paths:
            try:    
                np.savez(
                    c_path,
                    readout=data["readout"],
                    readout_shape=data["readout_shape"],
                    readout_method=data["readout_method"],
                    simulation_config=data['simulation_config'],
                )
                print(f"Saving - {c_path}")
                full_save_path_and_name = c_path
                break
            except:
                errors.append(c_path)

        if len(errors)>0:
            print("Saving incurred errors:")
            for e,path in errors:
                print(f"{e} : {path}")

        
        if isinstance(data["readout"][0],np.memmap) and cleanup_tmps:
            print("Attempting to clean up tempory files... ",end="")
            try:
                for readout in data["readouts"]:
                    readout.flush()
                    tmp_filename = readout.filename
                    error = 0
                    if os.path.exists(tmp_filename):
                        os.remove(tmp_filename)
                    else: error+=1
                    
                    if not error:
                        print("Success!")
                    else:
                        print(f"\n({error} Errors) A tempory file was moved or already deleted. This is either good news or terrible news (joke).")
                data["readouts"].clear()
                gc.collect()
            except Exception as e:
                print(f"{e} : Temporary file cleanup failed due to unknown error.")

        return full_save_path_and_name

    def save_pca(self,data,prefix=""):
        assert os.path.exists(self.save_path), f"{self.save_path} - Save path no longer exists."  
        if prefix=="": 
            name = f"{os.path.basename(self.save_path)}_pcas"
        else:
            if prefix.isalnum():
                name = f"{prefix}_pcas"
            else:
                raise ValueError(f"Suffix {prefix} is an invalid file name. Must be alpha numeric only.")
        main_save_path = self._unique_name(name) 
        name = os.path.join(self.pcas_dir_name,name)
        subdir_save_path = self._unique_name(name) 

        candidate_paths = []
        candidate_paths.append(subdir_save_path)
        candidate_paths.append(main_save_path)

        full_save_path_and_name = None
        errors = []
        for c_path in candidate_paths:
            try:    
                np.savez(
                    c_path,
                    methodology=str(data["methodology"]),
                    consistent_capacity=data["consistent_capacity"],
                    consistency_profile=data["consistency_profile"],
                    simulation_config=data['simulation_config'],
                )
                print(f"Saving - {c_path}")
                full_save_path_and_name = c_path
                break
            except Exception as e:
                print(f"Path Failed - {c_path}")
                errors.append(e)

        if len(errors)==2:
            print("Saving incurred errors:")
            for path in errors:
                print(f"\t{path}")
            raise Exception("Save Failed.")

        return full_save_path_and_name



def load_run(path: str, memory_map=False):
    run_dict = {}
    with np.load(path, allow_pickle=True) as npz:
        run_dict["simulation_steps"] = int(npz.get("simulation_steps"))
        run_dict["boid_count"] = int(npz.get("boid_count"))
        run_dict["bounds"] = npz.get("bounds")
        run_dict["config"] = npz.get("config")

        simulation_steps = run_dict["simulation_steps"] 
        boid_count = run_dict["boid_count"]

        if memory_map:
            #inform the user what the f is going on

            #custom data structure so that str keying can still happen
            data_structure = np.dtype([
                ('positions', 'float64', (boid_count, 2)),
                ('velocities', 'float64', (boid_count, 2)),
                ('predator_positions', 'float64', (2,))
            ])

            #creating a tempory npy file
            basename = os.path.basename(path).split('.')[0]
            prefix = f"{basename}_"
            os.makedirs(TMP_PATH,exist_ok=True)
            npy_file = tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix='.npy', dir=TMP_PATH)
            npy_path = npy_file.name
            npy_file.close()

            print(f"Made tmp file at {npy_path}")

            #create memmap object
            npy = np.memmap(npy_path, dtype=data_structure, mode='w+', shape=(simulation_steps,))

            #save the big parts of the npz to the disk (the tempory npy)
            #flush after each cos they might be huge
            npy['positions'] = npz.get("positions")
            npy.flush()
            npy['velocities'] = npz.get("velocities")
            npy.flush()
            npy["predator_positions"] = npz.get("predator_positions")
            npy.flush()


            #map the dictionary to the npy
            run_dict["positions"] = npy['positions']
            run_dict["velocities"] = npy['velocities']
            run_dict["predator_positions"] = npy['predator_positions']
            
            run_dict['memory_map'] = npy_path #truthy anyway
            
        else:
            run_dict["positions"]           = npz.get("positions")
            run_dict["velocities"]          = npz.get("velocities")
            run_dict["predator_positions"]  = npz.get("predator_positions")
            
        for k,v in run_dict.items():
            if type(v) is tuple:
                run_dict[k] = v[0]
            if v is None:
                print(f"\tvalue for '{k}' is missing, features relating to this will not work thus.")

    return run_dict


def load_pca(path:str,memory_map=False):
    pcas = {}
    npz = np.load(path, allow_pickle=True)
    with np.load(path, allow_pickle=True) as npz:
        pcas["methodology"] = npz.get("methodology")
        pcas["consistent_capacity"] = npz.get("consistent_capacity")
        pcas["consistency_profile"] = npz.get("consistency_profile")
        pcas["simulation_config"] = npz.get("simulation_config")
    return pcas


def load_prediction(path: str, memory_map=False):
    prediction_dict = {}

    with np.load(path, allow_pickle=True) as npz:
        prediction_dict["readout_method"] = npz["readout_method"]
        prediction_dict["prediction_distance"] = npz["prediction_distance"]
        prediction_dict["train_size"] = npz["train_size"]
        prediction_dict["alpha"] = npz["alpha"]
        prediction_dict["alpha_search"] = npz["alpha_search"]
        prediction_dict["corr_coef"] = npz["corr_coef"]
        prediction_dict["simulation_config"] = npz["simulation_config"]
        prediction_dict["y_test"] = npz["y_test"]


        if memory_map:
            prediction = npz["prediction"]
            prefix = f"{os.path.basename(path).split('.')[0]}_"
            mmap, mmap_path = create_mmap(prefix=prefix, shape=prediction.shape, dtype=prediction.dtype)

            mmap[:] = prediction
            mmap.flush()

            prediction_dict["prediction"] = mmap
            prediction_dict["memory_map"] = mmap_path

        else:
            prediction_dict["prediction"] = npz["prediction"]

    return prediction_dict


def load_readout(path: str, memory_map=False):
    readout_dict = {}

    with np.load(path, allow_pickle=True) as npz:
        readout_dict["readout_method"] = npz["readout_method"]
        readout_dict["simulation_config"] = npz["simulation_config"]
        readout_dict["readout_shape"] = tuple(npz["readout_shape"])

        if memory_map:
            readout = npz["readout"]
            prefix = f"{os.path.basename(path).split('.')[0]}_"
            mmap, mmap_path = create_mmap(prefix=prefix, shape=readout.shape, dtype=readout.dtype)

            mmap[:] = readout
            mmap.flush()

            readout_dict["readout"] = mmap
            readout_dict["memory_map"] = mmap_path
        else:
            readout_dict["readout"] = npz["readout"]

    return readout_dict


def load_npzs(paths: list,re_filter:str="",memory_map=False,load_function=load_run):
    """Used to load multiple NPZ files given a load function corresponding to the saved NPZs type

    Parameters
    ----------
    paths : list
        list of paths which should be searched for npz files
    re_filter : str, optional
        case insensitive name filter on the files. Finds matching patterns
    memory_map : bool, optional
        whether or not to use numpy memory mapping when loading the npzs, by default False
    load_function : callable, optional
        function used to load the npz files. Options are load_run, load_pca, load_prediction, load_readout. By default load_run

    Returns
    -------
    list
        list of dictionary forms of the npz files

    Raises
    ------
    ValueError
        if the load function isn't one of the supported ones
    """


    if load_function is None or not callable(load_function):
        raise ValueError(f"Load function invalid.")

    if isinstance(paths, str):
        paths = [paths]
        
    npz_paths = []
    for p in paths:
        if os.path.isdir(p):
            with os.scandir(p) as sub_files:
                for f in sub_files:
                    if f.name.endswith('.npz') and re.search(re_filter.lower(), f.name.lower()):
                        npz_paths.append(f.path)
        else:
            if re.search(re_filter.lower(), os.path.basename(p).lower()):
                npz_paths.append(p)

    if memory_map: 
        print(f"Using numpy's memory mapping:")

    datas = [load_function(f,memory_map) for f in npz_paths]

    print(f'\nLoaded {len(datas)} npz(s)')
    return datas




def create_mmap(prefix, shape, dtype='float64'):
    """
    Helper function for generating tempory `.npy` files which are used with memory mapping and chunking.

    - Makes a temporary dir `./tmp`
    - Creates temporary file in this directory

    Parameters
    ----------
    prefix : str
        Natural language prefix for the temporary file name in the pattern `PREFIX + UUID + .npy`.
        As to say, the user doesn't need to worry about ensuring unique names, this is done automatically.
    shape : list[ints]
        shape of the created matrix inside the np.memmap.
    dtype : str, optional
        datatype of the np.memmap, by default 'float64'.

    Returns
    -------
    np.memmap
        np.memmap object.
    str
        path to temporary file created.

    Notes
    -----
    `tempfile.NamedTemporaryFile(delete=False,...)` delete is false because the temporary files may be used for operations done in a jupyter notebook.
    """        

    os.makedirs(TMP_PATH, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix='.npy', dir=TMP_PATH)
    path = tmp_file.name
    tmp_file.close()
    print(f"Made tmp file at {path}")
    return np.memmap(path, dtype=dtype, mode='w+', shape=shape), path


def npy_cleanup(tmp_dir_path,re_filter:str=""):
    """
        Use to cleanup (delete) temporary .npy files.

    Parameters
    ----------
    tmp_dir_path : str
        path to directory containing files to be cleaned up (deleted)
    re_filter : str, optional
        optional regex filter for determining which files are deleted

    Raises
    ------
    FileNotFoundError
        when tmp_dir_path doesn't exist
    """

    paths = []
    if not isinstance(tmp_dir_path,list):
        paths.append(tmp_dir_path)
    else:
        paths = tmp_dir_path

    count = 0
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        if os.path.isdir(p):
            for f in os.scandir(p):
                if f.name.endswith('.npy') and re.search(re_filter, f.name):
                    try:
                        os.remove(f.path)
                        count += 1
                    except Exception as e:
                        print(f"Failed to remove {f.path}: {e}")
        else:
            filename = os.path.basename(p)
            if filename.endswith('.npy') and re.search(re_filter, filename):
                try:
                    os.remove(p)
                    count += 1
                except Exception as e:
                    print(f"Failed to remove {p}: {e}")

    print(f"Removed {count} temporary .npy file(s).")







def peterb_run(path: str):
    """
        This function takes a path to an npz and peterbs the starting conditions according to

    Parameters
    ----------
    path : str
        _description_
    """

