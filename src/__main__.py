from os.path import samefile
from numpy import sign
from tqdm import trange
import argparse
import os
import matplotlib.pyplot as plt

from .SaverLoader import Saver, find_npzs
from .BoidSimulator import BoidSimulator
from .BoidVisualiser import flat_render,torus_render
from .ConfigManager import load_config, generate_config, compare_params


parser = argparse.ArgumentParser()
parser.add_argument('--run',            '-r',   type=str,   nargs='?', const=True,                 help="Run a simulation given a .ini file path of parameters.")
parser.add_argument('--iterations',     '-i',   type=int,   nargs='?', default=1,                  help="Number of times you want the simulation to be run.")
parser.add_argument('--seed',                   type=int,  nargs='?', default=True, const=True,    help="Whether or not the simulator uses a random seed. False uses the seed from the config file.")
parser.add_argument('--generate-params','-g',   type=str,   nargs='?', const='./',                 help="Give a file path to generate an empty .ini with default parameters there.")
parser.add_argument('--view',           '-v',   type=str,   nargs='?', const=True,                 help="View a previous simulation, given the file path of a valid '.npz'. Defaults to true, which can be used to view a generated run.")
parser.add_argument('--save',           '-s',   type=str,   nargs='?', const=False,                help="If iterations is more than 1, use to specify the directory in which a run is saved. Otherwise it can be used to choose a specific save name")
parser.add_argument('--chunk',          '-c',   type=int,   nargs='?', default=0,  const=10000,    help="int value for chunk size used; enables use of numpy memory mapping.")

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
    chunking = args.chunk
    
    we_be_saving = save_path is not None
    


    if chunking<0:#case where user inputs something less than 0
        return ValueError(f"Chunk size must be a positive int greater than 0.")
    else:
        we_be_memmap = chunking > 0#memory mapping is enabled if the user entered something more than zero
        

    # Checks
    print(f"run: {run}\niterations {iterations}\nseed: {seed}\nsave: {save_path}\nchunking: {chunking}")
    if not we_be_saving and run: 
        input('\n--------- WARNING ---------\nNo save path specified so the run will not be saved (dry run) abort CTRL-C or any key to continue with dry run')


    #(1)
    if view and not run:
        datas = find_npzs(view)
        same,difference_str = compare_params(datas,["coord_system","sim_width"])
        if not same:
            raise Exception(f"The simulations loaded do not have the same coordinate_system or simulation_width. See parameter comparison table below:{difference_str}")
        
        replicas = BoidSimulator.construct_view_dict(datas)

        if datas[0]["coord_system"] == 'flat':
            animation = flat_render(replicas,overlay=True,animate=True)
        else:
            sw = datas[0]["sim_width"]
            animation = torus_render(replicas,sim_width=sw,overlay=True,animate=True)

        plt.show()
        return 

    #(2)
    if run:
        ini_path = run
        simulation_parameters = load_config(ini_path)
        if we_be_saving: 
            saver = Saver(save_path)

        # setting the seed
        Simulator = BoidSimulator(simulation_parameters,seed,memory_mapping=we_be_memmap,chunk_size=chunking)

        datas=[]
        for i in trange(iterations,desc='Iterations',position=0):
            result = Simulator.run_simulation()
            datas.append(result) 
            if we_be_saving: saver.save_run(result)


        if view: 
            replicas = BoidSimulator.construct_view_dict(datas)
            if simulation_parameters["coord_system"] == 'flat':
                animation = flat_render(replicas,overlay=True,animate=True)
            else:
                sw = simulation_parameters["sim_width"]
                animation = torus_render(replicas,sim_width=sw,overlay=True,animate=True)
            plt.show()
        #TODO torus render not working yet
        return datas

    raise Exception('No arguments were given but parser help cant be printed')


if __name__ == "__main__":
    main()