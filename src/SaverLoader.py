import numpy as np
import os
import tempfile
import gc

TMP_PATH = 'tmp'

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
        candidate_path = os.path.join(self.save_path,f'{candidate_name}_run')

        while os.path.exists(candidate_path+str(iter)+extension):
            if isinstance(iter,str): 
                # this is for the first loop only
                iter = 0
            else: 
                iter+=1

        return candidate_path+str(iter)+extension

    
    def save_run(self, data):
        assert os.path.exists(self.save_path), f"{self.save_path} - Save path no longer exists."
        name = os.path.basename(self.save_path)
        full_save_path_and_name = self._unique_name(name) 

        try:
            np.savez(
                full_save_path_and_name,
                positions=data["positions"],
                velocities=data["velocities"],
                predator_positions=data["predator_positions"],
                simulation_steps=data["simulation_steps"],
                boid_count=data["boid_count"],
                bounds=data["bounds"],
                config=data['config'],
            )
        except InterruptedError:
            print("Couldn't save run!")


        print(f"Saving - {full_save_path_and_name}")

        if isinstance(data["positions"],np.memmap):
            print("Attempting to clean up tempory files... ",end="")
            tmp_pos_filename = data["positions"].filename
            tmp_vel_filename = data["velocities"].filename

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
            

        return full_save_path_and_name

    
    def _validate_save_path(self, path:str):
        if os.path.isdir(path):
            return os.path.abspath(path)
        else:
            raise FileNotFoundError(f"Directory {path} doesn't exist or cant be found.")



def find_npzs(paths: list,memory_map=False):
    '''
    Takes some paths, of directories (in which it searches for npzs, or npz paths)
    :return: array of dictionaries containing the npz runs it found
    '''
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

    datas = [load_run(f,memory_map) for f in npz_paths]

    print(f'\nLoaded {len(datas)} run(s)')
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

