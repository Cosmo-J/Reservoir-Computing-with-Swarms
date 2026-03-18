#this is a copy which attempts to use an external params manager
from tqdm import trange
from tqdm import tqdm
import argparse
import os
from concurrent.futures import ThreadPoolExecutor as TPE, as_completed
import matplotlib.pyplot as plt

from . import BoidSimulator, BoidVisualizer


parser = argparse.ArgumentParser()
parser.add_argument('--run',            '-r',   type=str,   nargs='?', const=True,  help="Run a new simulation given a .ini file path of parameters.")
parser.add_argument('--iterations',     '-i',   type=int,   nargs='?', default=1,   help="Use with run or runview to perform multiple simulations.")
parser.add_argument('--seed',                   type=int,   nargs='?', const=1,     help="Give a random seed to produce identical runs. Default seed is '1'.")
parser.add_argument('--generate-params','-g',   type=str,   nargs='?', const='./',   help="generate an empty .ini with default parameters.")
parser.add_argument('--view',           '-v',   type=str,   nargs='?', const=True,         help="View a previous simulation, given the file path of a valid '.npz'. Defaults to true, which can be used to view a generated run.")
parser.add_argument('--save',           '-s',   type=str,   nargs='?', const=False,  help="If iterations is more than 1, this parameter is taken as the save directory. Otherwise")
parser.add_argument('--multithread',    '-mt',   type=bool, nargs='?', default=False,  const=True, help="set true ")

DEFAULT_SAVE_PATH = 'tests/boid_runs'
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

    Simulator = BoidSimulator()
    ParamManager, SaverLoader = Simulator.initialise()
    # if generating params, early return because this is an iscolated use case
    if args.generate_params:
        out_path = f"{args.generate_params}default_params.ini"
        print(f'Generating new parameters file at {out_path}')
        ParamManager.write_default_ini(out_path)
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
        input('\n--------- WARNING ---------\nNo save path, run will not be saved (dry run) abort CTRL-C or any key to continue with dry run')


    #(1)
    if view and not run:
        datas = SaverLoader.find_npzs(view)
        bv = BoidVisualizer(datas)
        ani = bv.get_animation(overlay=True)
        plt.show()
        return 

    #(2)
    if run:
        ini_path = run
        ParamManager.load_params_from_ini(ini_path)
        if we_be_saving: SaverLoader.set_save_dir(save_path)

        # setting the seed
        Simulator.set_seed(seed)

        if multithread:
            datas = [None] * iterations
            executor = TPE(max_workers=4)
            futures = {executor.submit(Simulator.run_simulation):i for i in range(iterations)}
            for f in tqdm(as_completed(futures),total=len(futures),desc='Iterations',position=0):
                i = futures[f]
                result = f.result()
                datas[i] = result
                
                if we_be_saving: SaverLoader.save_run(result,config_title=run)
            executor.shutdown()
        else:
            datas=[]
            for i in trange(iterations,desc='Iterations',position=0):
                result = Simulator.run_simulation()
                datas.append(result) 
                if we_be_saving: SaverLoader.save_run(result,config_title=run)



        if view: 
            bv = BoidVisualizer(datas)
            ani = bv.get_animation(overlay=True)
            plt.show()
        
        return datas

    raise Exception('No arguments were given but parser help cant be printed')


if __name__ == "__main__":
    main()