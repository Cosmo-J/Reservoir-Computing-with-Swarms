import numpy as np
import os

class SimSaverLoader:
    def __init__(self,save_path='boid_runs',config_title = 'NaN'):
        if save_path[-1]=='/':
            save_path=save_path[:-1]

        self.save_path = save_path
        self.config_title = config_title

    def _unique_name(self,candidate_name:str):
        '''
        :candidate_name: this is the potential file name /path/to/file
        :return: Modified path /path/to/file_run_12.npz
        '''
        extension = '.npz'
        iter = ''
        candidate_path = f'{self.save_path}/{candidate_name}_run'
        while os.path.exists(candidate_path+str(iter)+extension):
            if isinstance(iter,str): 
                # this is for the first loop only
                iter = 0
            else: 
                iter+=1

        return candidate_path+str(iter)+extension

    def save_run(self, data,config_title:str = None):
        os.makedirs(self.save_path, exist_ok=True)
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

    def set_save_dir(self, path:str):
        """
            if the path is directory, name the runs the smallest number iter inside said directory
        """
        if os.path.isdir(path):
            self.save_path = path
        else:
            raise FileNotFoundError(f"Directory {path} doesn't exist or cant be found.")


    @staticmethod
    def find_npzs(paths: list):
        '''
        Takes some paths, of directories (in which it searches for npzs, or npz paths)
        :return: array of dictionaries containing the npz runs it found
        '''
        if isinstance(paths, str):
            paths = [paths]
            
        npzs = []
        for p in paths:
            if os.path.isdir(p):
                sub_files = os.scandir(p)
                npzs.extend([f.path for f in sub_files if f.name.endswith('.npz')])
            else:
                npzs.append(p)
        
        datas = [SimSaverLoader.load_run(f) for f in npzs]
        print(f'Loaded {len(datas)} run(s)')

        return datas

    @staticmethod
    def load_run( path: str) -> dict:
        z = np.load(path, allow_pickle=True)
        run_dict = {}
        run_dict["positions"] = z.get("positions"),
        run_dict["velocities"] = z.get("velocities"),
        run_dict["predator_positions"] = z.get("predator_positions"),
        run_dict["time_steps"] = int(z.get("time_steps")),
        run_dict["boid_count"] = int(z.get("boid_count")),
        run_dict["bounds"] = z.get("bounds"),
        run_dict["config_title"] = z.get("config_title"),
        run_dict["config"] = z.get("config")
        
        for k,v in run_dict.items():
            if type(v) is tuple:
                run_dict[k] = v[0]
            if v is None:
                print(f"\tvalue for '{k}' is missing, features relating to this will not work thus.")

        return run_dict
