import numpy as np
import os
import tempfile

TMP_PATH = 'tmp'

class SimSaverLoader:
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

    
    def save_run(self, data,config_title:str = None):
        assert os.path.exists(self.save_path), f"{self.save_path} - Save path no longer exists."
        name = os.path.basename(self.save_path)
        full_save_path_and_name = self._unique_name(name) 

        try:
            np.savez(
                full_save_path_and_name,
                positions=data["positions"],
                velocities=data["velocities"],
                predator_positions=data["predator_positions"],
                time_steps=data["time_steps"],
                boid_count=data["boid_count"],
                bounds=data["bounds"],
                config=data['config'],
                config_title = [config_title]
            )
        except InterruptedError:
            print("Couldn't save run!")

        print(f"Saving - {full_save_path_and_name}")
        return full_save_path_and_name

    
    def _validate_save_path(self, path:str):
        if os.path.isdir(path):
            return os.path.abspath(path)
        else:
            raise FileNotFoundError(f"Directory {path} doesn't exist or cant be found.")


    @staticmethod
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

        datas = [SimSaverLoader.load_run(f,memory_map) for f in npz_paths]

        print(f'\nLoaded {len(datas)} run(s)')
        return datas

    @staticmethod
    def load_run(path: str, memory_map=False):

        run_dict = {}
        npz = np.load(path, allow_pickle=True)
        run_dict["time_steps"]          = int(npz.get("time_steps"))
        run_dict["boid_count"]          = int(npz.get("boid_count"))
        run_dict["bounds"]              = npz.get("bounds")
        run_dict["config_title"]        = npz.get("config_title")
        run_dict["config"]              = npz.get("config")

        time_steps = run_dict["time_steps"] 
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
            npy = np.memmap(npy_path, dtype=data_structure, mode='w+', shape=(time_steps,))

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
