from tqdm import trange
import argparse
import sys
import os
import matplotlib.pyplot as plt

from .SaverLoader import Saver, load_npzs
from .Simulators import BoidSimulator, LorenzSimulator
from .Visualisers import swarm_render,torus_swarm_render
from .ConfigManager import load_config, generate_config, compare_params


parser = argparse.ArgumentParser()
parser.add_argument('--run',            '-r',   type=str,   nargs='?', const=True,                 help="Path to a .ini config file containing simulation parameters.")
parser.add_argument('--save',           '-s',   type=str,   nargs='?', const=False,                help="Path to directory where simulations are saved on completion.")
parser.add_argument('--view',           '-v',   type=str,   nargs='?', const=True,                 help="Path to an .npz containing simulation data. Animates the simulation.")
parser.add_argument('--iterations',     '-i',   type=int,   nargs='?', default=1,                  help="Number of times the simulation is run. Default is 1.")
parser.add_argument('--chunk',          '-c',   type=int,   nargs='?', default=0,                   help="Int value for chunk size used; enables use of numpy memory mapping.")
parser.add_argument('--no-seed',                type=int,  nargs='?', default=False, const=True,    help="Overide config seed with a random value.")
parser.add_argument('--generate-config','-g',   type=str,   nargs='?', const='./',                 help="Path to directory where a template .ini will be generated.")


DEFAULT_SAVE_PATH = 'tests/'
def main():
    os.makedirs(DEFAULT_SAVE_PATH, exist_ok=True)
    #below accounts for main being called by other scripts, where custom args is a custom object
    args = parser.parse_args()
    if not len(sys.argv) > 1:
        parser.print_help()
        sys.exit(1)


    # if generating params, early return because this is an iscolated use case
    if args.generate_config:
        target_dir = args.generate_config
        if target_dir == './':
            target_dir = DEFAULT_SAVE_PATH

        generate_config(target_dir)
        return


    view = args.view # view some number of previous runs (.nz) 
    run = args.run # do a run returning the result
    iterations = args.iterations # Called replicas because they have different start conditions
    no_seed = args.no_seed
    save_path = args.save # save a run (given run or runview)
    chunking = args.chunk
    
    we_be_saving = save_path is not None
    


    if chunking<0:#case where user inputs something less than 0
        return ValueError(f"Chunk size must be a positive int greater than 0.")
    else:
        we_be_memmap = chunking > 0#memory mapping is enabled if the user entered something more than zero
        

    # Checks
    print(f"run: {run}\niterations {iterations}\nseed: {no_seed}\nsave: {save_path}\nchunking: {chunking}")


    #(1)
    if view and not run:
        datas = load_npzs(view)
        same,difference_str = compare_params(datas,["coord_system","sim_width"])
        if not same:
            raise Exception(f"The simulations loaded do not have the same coordinate_system or simulation_width. See parameter comparison table below:{difference_str}")
        
        local_config = datas[0].get("config").item()
        if local_config["coord_system"] == 'flat':
            animation = swarm_render(datas,overlay=True,animate=True)
        else:
            sw = local_config["SIM_WIDTH"]
            animation = torus_swarm_render(datas,sim_width=sw,overlay=True,animate=True)

        plt.show()
        return 

    #(2)
    if run:
        ini_path = run
        simulation_parameters = load_config(ini_path)
        if we_be_saving: 
            saver = Saver(save_path)
        else: 
            input('\n--------- WARNING ---------\nRun will not be saved (dry run). Abort with CTRL-C, or press any key to continue...')

        # setting the seed

        # Generate the lorenz system
        lorenz_system = LorenzSimulator(simulation_parameters['l_sigma'],simulation_parameters['l_rho'],simulation_parameters['l_beta'])
        l_xy = lorenz_system.generate_lorenz(simulation_parameters['simulation_steps'], simulation_parameters['l_sampling_rate'], simulation_parameters['x_lorenz'], simulation_parameters['y_lorenz'], simulation_parameters['z_lorenz'])

        # Initialise Simualtor
        Simulator = BoidSimulator(simulation_parameters,driving_signal=l_xy,use_random_seed=no_seed,memory_mapping=we_be_memmap,chunk_size=chunking)

        datas=[]
        for i in trange(iterations,desc='Iterations',position=0):
            result = Simulator.run_simulation()
            datas.append(result) 
            if we_be_saving: 
                saver.save_run(result,cleanup_tmps=True)


        if view: 
            if simulation_parameters["coord_system"] == 'flat':
                animation = swarm_render(datas,overlay=True,animate=True)
            else:
                sw = simulation_parameters["sim_width"]
                animation = torus_swarm_render(datas,sim_width=sw,overlay=True,animate=True)
            plt.show()
        #TODO torus render not working yet
        return datas


if __name__ == "__main__":
    
    main()