import numpy as np
import os
import tempfile
import gc

TMP_PATH = 'tmp'
PREDICTIONS_DIR_NAME = "predictions"
RUNS_DIR_NAME = "runs"
READOUTS_DIR_NAME = "readouts"

class Saver:
    def __init__(self,save_path):
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
            sub_dir_names = [PREDICTIONS_DIR_NAME,RUNS_DIR_NAME,READOUTS_DIR_NAME]
            for d in sub_dir_names:
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
        name = os.path.join(RUNS_DIR_NAME,name)
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
            except Exception as e:
                errors.append((e,c_path))


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
        name = os.path.join(PREDICTIONS_DIR_NAME,name)
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
        name = os.path.join(READOUTS_DIR_NAME,name)
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
            except Exception as e:
                errors.append((e,c_path))

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


def load_npzs(paths: list,memory_map=False,npz_type="run"):
    '''
        Takes some paths, of directories (in which it searches for npzs, or npz paths)
        :return: array of dictionaries containing the npz runs it found
    '''
    npz_type = npz_type.strip().lower()
    if npz_type not in ["run","prediction","readout"]:
        raise ValueError(f"Parameter npz_type must be either 'run' or 'prediction'")

    if isinstance(paths, str):
        paths = [paths]
        
    npz_paths = []
    for p in paths:
        if os.path.isdir(p):
            sub_files = os.scandir(p)
            npz_paths.extend([f.path for f in sub_files if f.name.endswith('.npz')])
        else:
            npz_paths.append(p)
    
    if memory_map: 
        print(f"Using numpy's memory mapping:")

    if npz_type == "run":
        datas = [load_run(f,memory_map) for f in npz_paths]
    elif npz_type == "prediction":
        datas = [load_prediction(f,memory_map) for f in npz_paths]
    elif npz_type == "readout":
        datas = [load_readout(f,memory_map) for f in npz_paths]

    print(f'\nLoaded {len(datas)} {npz_type}(s)')
    return datas


def load_run(path: str, memory_map=False):
    run_dict = {}
    npz = np.load(path, allow_pickle=True)

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
            readouts = npz["readouts"]
            mmap_readouts = []
            mmap_paths = []

            for i,r_shape in enumerate(readout_dict["readout_shape"]):
                name = f"{os.path.basename(path).split('.')[0]}_readout_{i}_"
                mmap, mmap_path = create_mmap(prefix=name,shape=r_shape)

                mmap[:] = readouts[i]
                mmap.flush()

                mmap_readouts.append(mmap)
                mmap_paths.append(mmap_path)

            readout_dict["readouts"] = mmap_readouts
            readout_dict["memory_map"] = mmap_paths

        else:
            readout_dict["readouts"] = npz["readouts"]

    return readout_dict


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


def cleanup_tmps(paths):
    failed_paths = []
    for p in paths:
        try:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    failed_paths.append((e,p))
        except Exception as e:
            failed_paths.append((e,p))

    print(f"Succesfully removed {len(paths)-len(failed_paths)} temporary files.")
    if len(failed_paths)!=0:
        print(f"Failed to remove {len(failed_paths)}. Exceptions:")
        for e,p in failed_paths:
            print(f"\t{e}: {p}")



#TODO write these functions
def mmap_cleanup(ObjectType,tmp_path=TMP_PATH):
    #TODO make this function work
        """
            Called internally if class instance is initialised with `cleanup_tmps=True`
        
            - Find a list of temporary files by looking inside `./tmp`.
            - Goes through live instances of the ObjectType objects, and subtracts any tempfile references from the aformentioned list.
            - Deletes the remaining tmp files in the list.
            - Uses gc.get_objects which can be slow
        """        
        if not os.path.exists(TMP_PATH): return
        files_in_tmp = os.scandir(TMP_PATH)
        tmp_files_paths = {os.path.abspath(f.path) for f in files_in_tmp if f.name.endswith('.npy')}

        gc.collect()
        found_refs = set()
        found_refs.update(tmp_path)
        
        for obj in gc.get_objects():
            if isinstance(obj, ObjectType):
                for path in obj.tmp_paths:
                    found_refs.add(path)

        orphans = tmp_files_paths - found_refs

        count = 0
        for orphan_path in orphans:
            try:
                os.remove(orphan_path)
                count += 1
            except OSError:
                print(f"Failed to remove tmp file {orphan_path}, either because its being referenced somewhere or locked.")

        if count > 0:
            print(f"Cleaned up (deleted) {count} orphaned temporary file(s).")


def peterb_run(path: str):
    """
        This function takes a path to an npz and peterbs the starting conditions according to

    Parameters
    ----------
    path : str
        _description_
    """

