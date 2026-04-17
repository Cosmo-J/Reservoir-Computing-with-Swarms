#this is a copy which attempts to use an external params manager
from tqdm import trange
from tqdm import tqdm
import argparse
import os
from concurrent.futures import ThreadPoolExecutor as TPE, as_completed
import matplotlib.pyplot as plt

from .SimSaverLoader import SimSaverLoader
from .BoidSimulator import BoidSimulator
from .BoidVisualizer import BoidVisualizer
from .ConfigManager import load_config, generate_config


parser = argparse.ArgumentParser()
parser.add_argument('--run',            '-r',   type=str,   nargs='?', const=True,                 help="Run a simulation given a .ini file path of parameters.")
parser.add_argument('--iterations',     '-i',   type=int,   nargs='?', default=1,                  help="Number of times you want the simulation to be run.")
parser.add_argument('--seed',                   type=bool,   nargs='?',default=True, const=True,  help="Whether or not the simulator uses a random seed. False uses the seed from the config file.")
parser.add_argument('--generate-params','-g',   type=str,   nargs='?', const='./',                 help="Give a file path to generate an empty .ini with default parameters there.")
parser.add_argument('--view',           '-v',   type=str,   nargs='?', const=True,                 help="View a previous simulation, given the file path of a valid '.npz'. Defaults to true, which can be used to view a generated run.")
parser.add_argument('--save',           '-s',   type=str,   nargs='?', const=False,                help="If iterations is more than 1, use to specify the directory in which a run is saved. Otherwise it can be used to choose a specific save name")
parser.add_argument('--multithread',    '-mt',   type=bool, nargs='?', default=False,  const=True, help="When running multiple iterations, set true to enable multithreading")

DEFAULT_SAVE_PATH = 'tests/'
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
        target_dir = args.generate_params
        if target_dir == './':
            target_dir = DEFAULT_SAVE_PATH

        generate_config(target_dir)
        return


    view = args.view # view some number of previous runs (.nz) 
    run = args.run # do a run returning the result
    iterations = args.iterations # Called replicas because they have different start conditions
    seed = args.seed
    save_path = args.save # save a run (given run or runview)
    multithread = args.multithread
    
    we_be_saving = save_path is not None


    # Checks
    print(f"run: {run}\niterations {iterations}\nseed: {seed}\nsave: {save_path}\nmultithreading: {multithread}")
    if not we_be_saving and run: 
        input('\n--------- WARNING ---------\nNo save path specified so the run will not be saved (dry run) abort CTRL-C or any key to continue with dry run')


    #(1)
    if view and not run:
        datas = SimSaverLoader.find_npzs(view)
        bv = BoidVisualizer(datas)
        ani = bv.get_animation(overlay=True)
        plt.show()
        return 

    #(2)
    if run:
        ini_path = run
        simulation_parameters = load_config(ini_path)
        if we_be_saving: 
            saver = SimSaverLoader(save_path)

        # setting the seed
        Simulator = BoidSimulator(simulation_parameters,seed)

        if multithread:
            datas = [None] * iterations
            executor = TPE(max_workers=4)
            futures = {executor.submit(Simulator.run_simulation):i for i in range(iterations)}
            for f in tqdm(as_completed(futures),total=len(futures),desc='Iterations',position=0):
                i = futures[f]
                result = f.result()
                datas[i] = result
                
                if we_be_saving: saver.save_run(result,config_title=run)
            executor.shutdown()
        else:
            datas=[]
            for i in trange(iterations,desc='Iterations',position=0):
                result = Simulator.run_simulation()
                datas.append(result) 
                if we_be_saving: saver.save_run(result,config_title=run)



        if view: 
            bv = BoidVisualizer(datas)
            ani = bv.get_animation(overlay=True)
            plt.show()
        
        return datas

    raise Exception('No arguments were given but parser help cant be printed')


if __name__ == "__main__":
    main()