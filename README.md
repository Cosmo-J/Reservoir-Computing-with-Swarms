Created in effort to recreate the methods of [Lymburn et al 2021](https://research-repository.uwa.edu.au/en/publications/reservoir-computing-with-swarms/).

This Python package provides a framework for simulating, visualising, and analyzing **Boid Swarm Dynamics**, specifically integrated with **Lorenz System** interactions where a predator follows chaotic trajectories. It also includes advanced tools for **Reservoir Computing** analysis, allowing for the prediction of chaotic systems based on swarm states.

---

## Overview
This package is designed to simulate boid agents in 2D (flat) or 3D (torus-mapped) environments. It facilitates:
* **Simulation**: Multi-threaded boid simulations with configurable forces (alignment, repulsion, homing, and predator avoidance).
* **Simulation Configurability**: configure simulation parameters using .ini files for the easy recreation of test cases.
* **Visualisation**: Interactive 2D and 3D plotting with playback controls.
* **Analysis**: Evaluation of the swarm as a "reservoir" for predicting external chaotic signals (like the Lorenz system).
* 

## Package Structure
* **`BoidSimulator`**: The core physics engine handling boid movements and force calculations.
* **`BoidVisualizer`**: A Matplotlib-based tool for animating simulation results in real-time or from saved files.
* **`SimParams`**: Used for creating and loading .ini file responsible for defining the simulation parameters.
* **`SimSaverLoader`**: Handles saving simulation data to `.npz` format and retrieving previous runs.
* **`ObservationLayer`**: Provides readout mechanisms (Kernels, Flat, COM) to transform swarm states into usable data for prediction tasks.

---

## Getting Started

### Installation
Ensure you have the following dependencies installed:
```bash
pip install -r requirements.txt
```

### Using the package
`python -m src -h` produces the help menu

```
usage: __main__.py [-h] [--run [RUN]] [--iterations [ITERATIONS]]
                   [--seed [SEED]] [--generate-params [GENERATE_PARAMS]]
                   [--view [VIEW]] [--save [SAVE]]
                   [--multithread [MULTITHREAD]]

options:
  -h, --help            show this help message and exit
  --run [RUN], -r [RUN]
                        Run a simulation given a .ini file path of parameters.
  --iterations [ITERATIONS], -i [ITERATIONS]
                        Number of times you want the simulation to be run.
  --seed [SEED]         Give a random seed to produce identical runs. Default
                        seed is '1'.
  --generate-params [GENERATE_PARAMS], -g [GENERATE_PARAMS]
                        generate an empty .ini with default parameters.
  --view [VIEW], -v [VIEW]
                        View a previous simulation, given the file path of a
                        valid '.npz'. Defaults to true, which can be used to
                        view a generated run.
  --save [SAVE], -s [SAVE]
                        If iterations is more than 1, use to specify the
                        directory in which a run is saved. Otherwise it can be
                        used to choose a specific save name
  --multithread [MULTITHREAD], -mt [MULTITHREAD]
                        When running multiple iterations, set true to enable
                        multithreading

```


#### Initialising Test Cases
To run a simulation, you need to specify a .ini file from which to load the parameters of said simulation. The command below can be used to generate default_params.ini

The command will also generate a tests folder, and a default_test_case directory within which, default_params.ini will be created
```bash
python -m src --generate-params ./
```
These config files can be renamed to anything you like.
The intended use is that different tests cases can be outlined with different config .ini files.

See an example file structure below which shows different folders representing different test cases with their corresponding ini files (arbitrarily named).  
*See also .npz runs produced from these .ini files*
```
tests
├── donut
│   └── donut.ini
│   ├── tests_run.npz
│	└── tests_run0.npz
└── default_test_case
    └── default_params.ini
    ├── tests_run.npz
	└── tests_run0.npz
```

#### Running Simulations
*a .ini as described in the previous section is required for running a simulation!*

##### Dry Run
To run a simulation you can use the following command:
```bash
python -m src --run PATH_TO_INI/default_params.ini
```
Since neither --view or --save have been specified above, it counts as a dry run which can be used to confirm that no errors have occurred in running the simulation.

##### Running and Saving
The `--save` parameter can be used to specify a save path where the run will be saved as a .npz file. Bear in mind that these files can get very large!
```bash
python -m src --run PATH_TO_INI/default_params.ini --save SAVE_PATH/
```

Later sections go over loading .npz runs so that further analysis can be performed.
##### Multiple Runs
Multiple runs can be conducted using the `--iterations` or '-i' parameter. 

The command below shows a command which will run 10 simulations based on `default_params.ini` in turn and save them all to the specified save path.
```bash
python -m src --run PATH_TO_INI/default_params.ini --save SAVE_PATH/ -i 10
```

Multithreading is supported and can be used to run multiple iterations concurrently. Although it's a fairly adhoc addition and I haven't made any effort to verify the difference it makes/modify the code so that it can be properly utilised. All to say, in my case, it doesn't seem to make a huge difference for the overall runtime with multiple iterations. Additionally, currently only quad threading is supported.

The example command below shows multithreading `-mt` being used in aid of generating 10 iterations of a run.
```bash
python -m src --run PATH_TO_INI/default_params.ini --save SAVE_PATH/ -i 10 -mt
```
##### Viewing
Runs can be viewed immediately after generation or later by specifying an .npz

Below is the same command from the previous section besides for the `--view` at the end, which means when the run is complete, a popup appears where the simulation can be viewed.
```bash
python -m src --run PATH_TO_INI/default_params.ini --save SAVE_PATH/ --view
```

You can also use `-v` to view saved runs
```bash
python -m src --view PATH_TO_SAVED_RUN/tests.npz
```
Or on a directory containing multiple .npz files to view multiple at once
```bash
python -m src --view DIR_CONTAINING_RUNS/
```

Viewing is very versatile. It can be used:
- when no save path is specified
`python -m src --run PATH_TO_INI/default_params.ini --view`
- when multiple iterations have been specified
`python -m src --run PATH_TO_INI/default_params.ini --view -i 10`

In the cases where viewing is conducted on more than one run, the runs are overlayed and given different colours.

#### Loading Simulations
Loading runs is of interest when wanting to perform further analysis on the results. It can be done by loading one of the included packages:
```python
from src import SimSaverLoader
datas = SimSaverLoader.load_runs(['path/to/run1.npz', 'path/to/run2.npz'])
```

datas is a list of dictionaries corresponding to the loaded runs. See the dictionary structure below:

| **Key**              | **Type**     | **Description**                                                                                         |
| -------------------- | ------------ | ------------------------------------------------------------------------------------------------------- |
| `positions`          | `np.ndarray` | Shape `(T, N, D)`. The coordinates of all $N$ boids over $T$ timesteps in $D$ dimensions.               |
| `velocities`         | `np.ndarray` | Shape `(T, N, D)`. The velocity vectors for all boids.                                                  |
| `predator_positions` | `np.ndarray` | Shape `(T, D)`. The trajectory of the Lorenz predator.                                                  |
| `time_steps`         | `int`        | The total duration of the simulation.                                                                   |
| `boid_count`         | `int`        | The number of agents ($N$) in the swarm.                                                                |
| `bounds`             | `tuple`      | The spatial boundaries/spawn area used in the simulation.                                               |
| `config`             | `dict`       | A nested dictionary containing every parameter from the `.ini` file used to generate this specific run. |
| `config_title`       | `str`        | The name of the parameter file or setup used for the run.                                               |
Note that some of these dictionary items are fairly redundant and only exist for conveniences sake. Moreover, `config` and `config_title` are saved to protect against the case where a user forgets the .ini used to produce a run.


#### Analysing Simulations
Simulations can be analysed using `ObservationLayer` which defines classes for getting readouts from a reservoir. Included is `FlatReadout` and `KernelReadout` which are different classes for reading the reservoir described in [Lymburn et al 2021](https://research-repository.uwa.edu.au/en/publications/reservoir-computing-with-swarms/). Additionally, the a superclass `ObservationAndPrediction` is included from which the two aforementioned readout classes extend. This exists for users who may want to define their own subclass which defines a specific methodology for reading reservoir states.

Below shows some examples of these classes being used:
```python
run_path = '/PATH_TO_RUNS'

datas = SimSaverLoader.find_npzs(run_path)
kr = KernelReadout(200,datas[0],datas[1],washout=1000)

sv1 = kr.get_reservoir_state_vectorised(kr.replica1)
sv2 = kr.get_reservoir_state_vectorised(kr.replica2)

cc, g2=kr.calc_consistency_profile_v1(sv1,sv2)

kr.plot_consistency_profile(cc,g2,truncated_to=100)

prediction,corr_coef = kr.ridge_prediction(sv2,prediction_distance=8)

plot = kr.get_plot(prediction,corr_coef,[0,2000])

plot.set_ybound(-7,7)
plt.show()
```
