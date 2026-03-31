import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

from .ConfigManager import ConfigManager

from tqdm import trange,tqdm
from concurrent.futures import ThreadPoolExecutor as TPE, as_completed
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial import KDTree
import os
import tempfile
import gc

TMP_PATH = 'tmp'

class ObservationAndPrediction(ABC):
    """
        Abstract Base Class for manging reservoir replicas, creating readouts on those replicas, 
        and performing the different kinds of analysis on those readouts found in Lymburn et al (2021).

        Only support analysis between two replicas of a simulation.

        See Also
        --------
        `KernelReadout` : Subclass which is used to generate observation kernels and perform kernel readouts.
        `NaiveReadout` : Subclass which is used to perform a naive readout.
        `COMReadout` : Subclass which is used to perform a center of mass readout.



        Parameters
        ----------
        replica1 : dict
            Data dictionary for a pre-simulated replica. Assumed to have been created using other parts of the package such as `BoidSimulator`, `SimSaverLoader`.
        replica2 : dict
            Expected to be identical to `replica1` in terms of:

            - time steps (length of simulation).
            - simulation parameters.
            - driving signal.

            Although expected that the reservoir had different starting conditions (positions/velocities).
        washout : int
            The number of initial time steps to discard to remove transient 
            dynamics.
        chunk_size : int, optional
            The number of rows processed per iteration during memory-mapped 
            operations. Default is 5000.
        cleanup_tmps : bool, optional
            If True, on initialisation, the class instance will search a tmp directory for temporary files used 
            in previous memory mapped operations which no longer have a live object which references them (orphaned).
            Note that if you're using python notebooks you may have to manually delete previous class instances `del(INSTANCE_NAME)`
            as overwriting a instance name doesn't remove them from the stack.

        Attributes
        ----------
        memory_map : bool
            Indicates whether the input data uses np.memmapping (memory mapping). Is checked by many other internal functions
            to determine whether to create and use temporary `.npy` files and chunking as to offload memory usage.
        tmp_paths : list[str]
            When a temporary file is created its path is added to this list. Referenced in `_cleanup_tmps` to create a whitelist 
            of still referenced temporary files that shouldn't be deleted.
        lorenz : np.ndarray
            Shape (N,2) N time_steps/samples and x,y position of the predator positions found in replica1. Intended usage is with its
            namesake a lorenz attractor, although in theory could be any driving signal stored as the predator positions in a given
            replica so long as it's shape is the same. Note, that the class does not check whether replica1 and replica2 have the same 
            driving signal, as this attribute is also used in calculations involving replica2.
    """

    def __init__(self,replica1,replica2,washout,chunk_size=5000,cleanup_tmps=True):
        self.replica1 = replica1
        self.replica2 = replica2
        self.washout_data(washout)
        self.lorenz = replica1.get('predator_positions')

        same_params, table = ConfigManager.compare_params(replica1.get('config').item(),replica2.get('config').item())
        assert same_params==True, "Replica1 and Replica2 have different parameters so are likely not replicas!:\n"+table
        assert np.allclose(replica1.get('predator_positions'), replica2.get('predator_positions')), "Replicas have different predator positions"

        # stuff relating to very large simulations
        rep1_memmap = replica1.get('memory_map',False)
        rep2_memmap = replica2.get('memory_map',False)

        self.memory_map = bool(rep1_memmap or rep2_memmap)
        self.chunk_size=chunk_size
        
        self.tmp_paths = []

        if self.memory_map:
            self.tmp_paths.append(rep1_memmap)
            self.tmp_paths.append(rep2_memmap)

        if cleanup_tmps: self._cleanup_tmps()


    def _cleanup_tmps(self):
        """
            Called internally if class instance is initialised with `cleanup_tmps=True`
        
            - Find a list of temporary files by looking inside `./tmp`.
            - Goes through live instances of the ObservationAndPrediction objects, and subtracts any tempfile references from the aformentioned list.
            - Deletes the remaining tmp files in the list.
            - Uses gc.get_objects which can be slow
        """        
        if not os.path.exists(TMP_PATH): return
        files_in_tmp = os.scandir(TMP_PATH)
        tmp_files_paths = {os.path.abspath(f.path) for f in files_in_tmp if f.name.endswith('.npy')}

        gc.collect()
        found_refs = set()
        found_refs.update(self.tmp_paths)
        
        for obj in gc.get_objects():
            if isinstance(obj, ObservationAndPrediction):
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


    def washout_data(self,washout):
        """
        Modifies the data inside the replica dictionaries slicing them so that the data only starts after the washout period.

        Parameters
        ----------
        washout : int
            Number of time steps to slice/remove from the start of the simulation.

            Alternatively, the (time-1) you wish the simulation to start at 
        """        
        def wash(data):
            data['positions'] = data['positions'][washout:]
            data['velocities'] = data['velocities'][washout:]
            data['predator_positions']= data['predator_positions'][washout:]
            data['time_steps']= data.get('time_steps') - washout
            return data
        
        self.replica1 = wash(self.replica1)
        self.replica2 = wash(self.replica2)


    @abstractmethod
    def get_reservoir_state_vectorised(self,replica):
        """
        Abstract function definition. Ensures that subclasses have a way of converting their replicas into vectorised readouts.

        Parameters
        ----------
        replica : self.replica#
            Class instance expects to be passed one of its own attibutes, either replica1 or replica2.

        Returns
        ------
        Intended to return some `np.ndarray` of shape (N,F) N time_steps/samples F features, for compatability with other methods in the class.
        """        
        pass


    @abstractmethod
    def calc_consistency_profile(self,sv1,sv2,methodology):
        """
        Calculate the consistency profile of a reservoir readout as described in the appendix of Lymburn et al (2021). 
        Allows for the specific `methodology` of calculating the consistency profile to be specified in allowance of the fact
        that Lymburn et al's methodology can be interpreted in multiple ways.

        Parameters
        ----------
        sv1 : np.ndarray, shape(N,F)
            Vectorised reservoir state of replica1. Assumed to have been generated using `get_reservoir_state_vectorised()`. 
            Numpy array with shape N time_steps/samples and F features/modes.
        sv2 : np.ndarray, shape(N,F)
            Same as sv1 except for replica2.
        methodology : str
            `function.__name__` reference of a given methodology for calculating the consistency profile.

        Returns
        -------
        consistent_capacity : float
            
            - theta 
            - consistent capacity
            - trace of the cross-covariance of `sv1` and `sv2`

        gamma_sqaured : list[floats]

            - eigenvalues of the cross-covariance of `sv1` and `sv2`
        """        
        pass


    @staticmethod
    def _create_mmap(prefix, shape, dtype='float64'):
        """
        Internal helper function for generating tempory `.npy` files which are used with memory mapping and chunking.

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
        print(f"\nMade tmp file at {path}")
        return np.memmap(path, dtype=dtype, mode='w+', shape=shape), path


    def _mean(self,sv):
        """
        Internal helper function for calculating the mean of a state vector. In other words, the average value of each feature across time.

        Parameters
        ----------
        sv : np.ndarray, shape(N,F)

            N time_steps/samples, F features/modes.

            Vectorised readout of a given replica across time.
        
        Returns
        -------
        np.ndarray or np.memmap, shape(F,)
            Numpy array with each mode time averaged.

        Notes
        -----
        The usefulness of this function is that, depending on the class instance's `memory_map` attribute it automatically determines whether
        to find the mean:
        
        - using chunks and `.npy` files.
        - using a single line operation `np.mean(sv,axis=0)`.

        Intended to be used inside different consistency profile methodologies to protect
        against overloading one's RAM given a very long time series replica.
        """        
        if self.memory_map:
            time_steps,features = sv.shape
            total_sum = np.zeros(features, dtype='float64')

            for chunk_start in trange(0,time_steps,self.chunk_size,desc='Chunked Mean'):
                chunk_end = min(chunk_start+self.chunk_size, time_steps)
                
                sv_chunk = sv[chunk_start:chunk_end]

                chunk_sum = np.sum(sv_chunk,axis=0)

                total_sum += chunk_sum

            return total_sum / time_steps
        else:
            return np.mean(sv,axis=0)


    def _center(self,sv):
        """
        Centers a given vector so that it has a zero mean. Similar to `_mean()` with the addition of subtracting the mean from the vector `sv1` feature wise.

        Parameters
        ----------
        sv : np.ndarray, shape(N,F)

            N time_steps/samples, F features/modes.

            Vectorised readout of a given replica across time.

        Returns
        -------
        np.ndarray or np.memmap, shape(N,F)
            same shape as the given vector except features now have a zero mean.

        Notes
        -----
        The usefullness of this function is that, depending on the class instance's `memory_map` attribute it automatically determines whether
        to find the center:
        
        - using chunks and `.npy` files.
        - using a single line operation `sv - np.mean(sv,axis=0)`.

        Intended to be used inside different consistency profile methodologies to protect
        against overloading one's RAM given a very long time series replica.
        """        
        if self.memory_map:
            time_steps,features = sv.shape
            total_sum = np.zeros(features,dtype='float64')

            for chunk_start in trange(0,time_steps,self.chunk_size,desc='Chunked Mean'):
                chunk_end = min(chunk_start+self.chunk_size,time_steps)
                sv_chunk = sv[chunk_start:chunk_end]
                chunk_sum = np.sum(sv_chunk,axis=0)
                total_sum += chunk_sum

            sv_mean = total_sum / time_steps

            sv_centered_npy, npy_path = self._create_mmap('sv_centered_',(time_steps,features))
            self.tmp_paths.append(npy_path)

            for chunk_start in trange(0,time_steps,self.chunk_size,desc='Chunked Mean'):
                chunk_end = min(chunk_start+self.chunk_size,time_steps)
                sv_chunk = sv[chunk_start:chunk_end]
                chunk_centered = sv_chunk-sv_mean
                sv_centered_npy[chunk_start:chunk_end] = chunk_centered
                sv_centered_npy.flush()

            return sv_centered_npy
        else:
            return sv - np.mean(sv,axis=0)


    def _sv_transform(self,sv,transform):
        """
        Apply a linear transform to state vector. 

        Parameters
        ----------
        sv : np.ndarray, shape(N,F)
            State vector of N time_steps/samples and F features/modes.
        
        transform : np.ndarray, shape(F,F)
            The transformation matrix applied to `sv`.

        Returns
        -------
        np.ndarray or np.memmap, shape(N,F)
            returns the state vector transformed. 

        Notes
        -----
        The usefullness of this function is that, depending on the class instance's `memory_map` attribute it automatically determines whether
        to calculate the transform:
        
        - using chunks and `.npy` files.
        - using a single line operation `sv @ transform`.

        Intended to be used inside different consistency profile methodologies to protect
        against overloading one's RAM given a very long time series replica.

        """

        if self.memory_map:
            time_steps, features = sv.shape
            mat_mul_out, npy_path = self._create_mmap('mat_mul_', (time_steps, features))
            self.tmp_paths.append(npy_path)

            for chunk_start in trange(0,time_steps,self.chunk_size,desc="Chunked matrix multiplication"):
                chunk_end = min(chunk_start+self.chunk_size,time_steps)

                mat_mul_out[chunk_start:chunk_end] = sv[chunk_start:chunk_end] @ transform
        
            mat_mul_out.flush()
            return mat_mul_out
        else:
            return sv @ transform


    def _covariance(self,sv1,sv2=None,center=False):
        """
        Calculates the covariance between two state vectors.

        Parameters
        ----------
        sv1 : np.ndarray, shape(T,F)
            state vector with shape (T,F) T samples/timesteps and F features/modes.
        sv2 : np.ndarray, shape(T,F), optional
            state vector with shape (T,F) T samples/timesteps and F features/modes.

            - if sv2 is None, autocovariance is calculated using sv1.

        center : bool, optional

            - True: centers the state vectors using the values from internal function `_mean`.
            - False: calculates the covariance with the assumption that state vectors already have 0 mean.

        Returns
        -------
            np.ndarray, shape(F,F)

        Notes
        -----
        The usefullness of this function is that, depending on the class instance's `memory_map` attribute it automatically determines whether
        to calculate the covariance:
        
        - using chunks and `.npy` files.
        - using a few lines.

        Intended to be used inside different consistency profile methodologies to protect
        against overloading one's RAM given a very long time series replica.


        - Assumes that F isn't super large and so the returned np.ndarray can live in memory.
        """      

        if sv2 is None:
            sv2 = sv1
            if center:
                sv1_mean = self._mean(sv1)
                sv2_mean = sv1_mean
        else:
            if center:
                sv1_mean = self._mean(sv1)
                sv2_mean = self._mean(sv2)
        if not center:
            sv1_mean = np.zeros(sv1.shape[1],dtype='float64')
            sv2_mean = np.zeros(sv2.shape[1],dtype='float64')


        if self.memory_map:
            time_steps,feats1 = sv1.shape
            _, feats2 = sv2.shape

            covariances = np.zeros((feats1, feats2), dtype='float64')

            for chunk_start in trange(0,time_steps,self.chunk_size,desc="Chunked Covariance"):
                chunk_end = min(chunk_start+self.chunk_size, time_steps)

                sv1_chunk_centered = sv1[chunk_start:chunk_end] - sv1_mean
                sv2_chunk_centered = sv2[chunk_start:chunk_end] - sv2_mean

                covariances += (sv1_chunk_centered.T @ sv2_chunk_centered)

            return covariances / (time_steps-1)
        else:
            sv1_centered = sv1-sv1_mean
            sv2_centered = sv2-sv2_mean
            
            time_steps = sv1_centered.shape[0]
            return (sv1_centered.T @ sv2_centered) / (time_steps-1)


    def ridge_prediction(self,state_vector,train_size=0.6,prediction_distance=1,ridge_alpha=None):
        """
        Use ridge regression to make a prediction about the x position of the lorenz attractor using  a reservoir readout.

        Parameters
        ---------
            state_vector1 : np.ndarray
                A state vector of shape (N,F) N time_steps/samples, F features/modes.

            train_size : float, optional
                Given only one state_vector, the fraction of the data used for training. Defaults to 0.6.

            prediction_distance : int, optional
                Number of simulation steps into the future the ridge regression will attempt to fit. Defaults to 1.

            ridge_alpha : float, optional 
                Defaults to None in which case an optimal alphas is calculated using `sklearn.linear_model.RidgeCV`. Otherwise, give a value to manually set alpha.

        Raises
        ------
        ValueError
            if the number of time steps of the state vectors are less than or equal to the prediction distance

        Returns
        -------
        prediction : np.ndarray
            Array of shape (N) where N is (time_steps - prediction_distance).
        corr_coef : float
            The correlation coefficient found against the prediction.
        alpha : float
            The alpha found if `ridge_alpha=None` (RidgeCV), otherwise, returns the parameter `ridge_alpha`
        """        
        
        if state_vector.shape[0]<=prediction_distance:
            raise ValueError(f"Prediction distance {prediction_distance} is greater than the number of time steps {state_vector.shape[0]}")

        y = self.lorenz[prediction_distance:,0]
        new_total_time = len(y)
        X = state_vector[:new_total_time]

        split_idx = int(new_total_time * train_size)

        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        mew = np.mean(X_train,axis=0)
        sigma = np.std(X_train,axis=0)

        X_train_stand = (X_train - mew)/sigma
        X_test_norm = (X_test - mew)/sigma

        if ridge_alpha == None:
            alphas = np.logspace(-3, 8, 50)
            #alphas=np.logspace(-6, 2, 20)
            ridge = RidgeCV(alphas=alphas, cv=5)
            ridge.fit(X_train_stand, y_train)
            best_alpha = ridge.alpha_
        else:
            ridge = Ridge(alpha=ridge_alpha)
            ridge.fit(X_train_stand, y_train)
            best_alpha = ridge_alpha

        prediction = ridge.predict(X_test_norm)

        #equation 13 from Lymburn et al, cosin similarity
        time_steps = y_test.shape[0]
        numer = np.sum((prediction * y_test))/time_steps
        denom = np.sqrt(np.mean(prediction**2) * np.mean(y_test**2))

        corr_coef = numer/denom

        return prediction, corr_coef, best_alpha


    def plot_ridge_prediction(self,prediction,corr_coef,prediction_distance,x_range=None,pop_out=False):
        """
        Intended to be used on the outputs of `ridge_prediction()`.
        Plots the predicted lorenz x coordinates against the actual lorenz coordinates, as well as displaying the correlation coefficient.

        Parameters
        ----------
        prediction : np.ndarray
            Shape (N,) which is the predicted position of the lorenz attractor at each N time step. Intended to be used with `ridge_prediction()`.
        corr_coef : float
            float correlation coefficient which is displayed at the top of the plot.
        x_range : tuple[float,float], optional
            The range of simulation steps to be displayed on the plot. `None` by default which shows the whole range [0,N].
        pop_out : bool, optional
            Pops out the plot with media esque controls for scrolling through and adjusting the displayed range. More for user analysis opposed to a shareable output. Defaults as `False`.

            - Requires `%matplotlib widget` in jupyter notebooks.

        Returns
        -------
        `~matplotlib.axes.Axes`
            returns an axes which you can add to other plots or just display on its own.
        
        See Also
        --------
            `ridge_prediction()` : For getting `prediction` and `corr_coef`
        """        
        lorenz_x_shifted = self.lorenz[prediction_distance:,0]
        prediction_start = len(lorenz_x_shifted) - len(prediction)
        lorenz_x = lorenz_x_shifted[prediction_start:]

        if x_range is None: 
            print("Plotting total range")
            x_range=[0,len(lorenz_x)]
        
        configs = self.replica1.get('config').item()
        sim_delta_t = configs['DELTA_T']
        look_ahead = prediction_distance*sim_delta_t


        y_label = f"lorenz_x(t+{look_ahead})"

        fig, ax = plt.subplots(figsize=(20, 6))
        plt.subplots_adjust(bottom=0.2)

        ax.plot(lorenz_x, color='red', label='lorenz_x')
        ax.plot(prediction, color='blue', linestyle='dashed', label='Prediction')
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)


        plt.xlabel('time')
        plt.ylabel(y_label)
        plt.title(f'correlation coefficient R: {corr_coef}')


        if pop_out:
            window_size = 2000
            max_index = len(lorenz_x)
            ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
            slider = Slider(ax_slider, 'Position', 0, max_index - window_size, valinit=0, valstep=10)

            ax_zoom = plt.axes([0.1, 0.01, 0.8, 0.03])
            zoom_slider = Slider(ax_zoom, 'Window Size', 100, max_index, valinit=window_size, valstep=100)

            def update(val):
                start = int(slider.val)
                end = start + int(zoom_slider.val)
                end = min(end, max_index)
                ax.set_xlim([start, end])
                fig.canvas.draw_idle()

            slider.on_changed(update)
            zoom_slider.on_changed(update)

            ax.set_xlim([0, window_size])
            plt.xlabel('simulation_steps')

        else:
            ax.set_xlim(x_range)
            ticks = ax.get_xticks()
            ax.set_xticks(ticks)#stupid line to stop matplotlib getting upset
            ax.set_xticklabels((ticks*sim_delta_t))

        return ax
    

    def plot_consistency_profile(self,consistent_capacity,gamma2_vector,truncated_to=100,cc_dp=1,show_consistent_capacity=True,dpi=200):
        """
        Used to plot the consistent capacity of a reservoir in the way of Lymburn et al (2021) figures 2b and 5b.

        Parameters
        ----------
        consistent_capacity : float
            consistent capacity as calculated by `calc_consistency_profile`.
        gamma2_vector : list[float]
            List of the gamma squared values, where each value is the consistency correlation of each mode.
        truncated_to : int, optional
            The number of modes or gamma squared values shown on the plot, by default 100 in parity with Lymburn et al (2021).
        cc_dp : int, optional
            the number of decimal places the consistent capacity is rounded to in the plot, by default 1
        show_consistent_capacity : bool, optional
            Whether to display the consistent capacity inside the plot, by default True.
        dpi : int, optional
            the resolution of the plot, dots per inch, by default 200.

        Returns
        -------
        `~matplotlib.axes.Axes`
            returns an axes which you can add to other plots or just display on its own.
        """        
        
        gamma2_k_ranked = np.sort(gamma2_vector,)[::-1]

        plt.rcParams['font.size'] = 18
        
        fig, ax = plt.subplots(dpi=dpi)
        
        ax.plot(gamma2_k_ranked[:truncated_to])
        ax.set_ylabel(r'$\gamma^{2}_{k}$',rotation=0,labelpad=20,fontsize=20)

        ax.set_xlabel(r'$k$')
        ax.set_xlim(left=0)
        ax.set_ybound([0,1])

        ax.set_xticks([0,truncated_to/2,truncated_to])
        ax.set_yticks([0,0.5,1])
        ax.set_box_aspect(1)
        fig.tight_layout()

        if show_consistent_capacity:
            middle_text = r'$\Theta$' + f'={np.round(consistent_capacity,decimals=cc_dp)}'
            ax.text(ax.get_xbound()[1]/2, ax.get_ybound()[1]/2, middle_text, fontweight='bold', horizontalalignment='center')   

        return ax


class KernelReadout(ObservationAndPrediction):
    """
        Subclass of `ObservationAndPrediction` used to generate observation kernels and perform kernel readouts.
        Heavily based on the kernel observation layer methodology described in Lymburn et al (2021).

        Only support analysis between two replicas of a simulation.

        See Also
        --------
        `ObservationAndPrediction` : Abstract Base Class for manging, creating readouts, and analysing replicas.
        `NaiveReadout` : Subclass which is used to perform a naive readout.
        `COMReadout` : Subclass which is used to perform a center of mass readout.


        Parameters
        ----------
        kernel_number : int
            The number of observation kernels that should be generated for the simulation.

        replica1 : dict
            Data dictionary for a pre-simulated replica. Assumed to have been created using other parts of the package such as `BoidSimulator`, `SimSaverLoader`.
        replica2 : dict
            Expected to be identical to `replica1` in terms of:

            - time steps (length of simulation).
            - simulation parameters.
            - driving signal.

            Although expected that the reservoir had different starting conditions (positions/velocities).
        washout : int
            The number of initial time steps to discard to remove transient 
            dynamics.
        chunk_size : int, optional
            The number of rows processed per iteration during memory-mapped 
            operations. Default is 5000.
        cleanup_tmps : bool, optional
            If True, on initialisation, the class instance will search a tmp directory for temporary files used 
            in previous memory mapped operations which no longer have a live object which references them (orphaned).
            Note that if you're using python notebooks you may have to manually delete previous class instances `del(INSTANCE_NAME)`
            as overwriting a instance name doesn't remove them from the stack.

        Attributes
        ----------
        memory_map : bool
            Indicates whether the input data uses np.memmapping (memory mapping). Is checked by many other internal functions
            to determine whether to create and use temporary `.npy` files and chunking as to offload memory usage.
        tmp_paths : list[str]
            When a temporary file is created its path is added to this list. Referenced in `_cleanup_tmps` to create a whitelist 
            of still referenced temporary files that shouldn't be deleted.
        lorenz : np.ndarray
            Shape (N,2) N time_steps/samples and x,y position of the predator positions found in replica1. Intended usage is with its
            namesake a lorenz attractor, although in theory could be any driving signal stored as the predator positions in a given
            replica so long as it's shape is the same. Note, that the class does not check whether replica1 and replica2 have the same 
            driving signal, as this attribute is also used in calculations involving replica2.
    """


    def __init__(self,kernel_number,replica1,replica2,washout,chunk_size):
        super().__init__(replica1,replica2,washout,chunk_size)
        self.kernel_number = kernel_number
        self.centers, self.widths = self.generate_kernels()


    def generate_kernels(self):
        """
        Generate a `self.kernel_number` number of kernels described by their center and width.

        Achieved using the methods outlined in Lymburn et al (2021). A kernel m has a center c_m and width w_m:

        - The position c_m is found by getting a random position of a boid at a random time.
        - The width w_m is the distance between said random boid and it's 5th closest neighber

        Returns
        -------
        np.ndarray : centers
            Shape (kernel_number, 2)
        
        np.ndarray : widths
            Shape(kernel_number,)
        """        

        positions = self.replica1['positions']
        time_steps = self.replica1['time_steps']
        boid_count = self.replica1['boid_count']
        
        widths = [None] * self.kernel_number
        centers = [None] * self.kernel_number

        for m in range(self.kernel_number):
            random_time = np.random.randint(0, time_steps) #selects a random t in the range of time steps of the simulation
            xs_at_random_time = positions[random_time] # selects the positions at the random time step

            random_agent_index = np.random.randint(0, boid_count)
            c_m = xs_at_random_time[random_agent_index] # some random agent position
            
            
            # "The width of the kernel is set to the distance to the 5th neighbor of the agent used to determine the location of the kernel at that time"
            tree = KDTree(xs_at_random_time) 
            distances, indices = tree.query(c_m, k=5)# find the closest 5 agents to random_agent, 6 
            w_m = distances[4]
            
            centers[m] = c_m
            widths[m]  = w_m
        

        return np.array(centers),np.array(widths)
    

    def get_reservoir_state_vectorised(self, replica):
        """
        This is implements the kernel observation layer described in Lymburn et al (2021).

        Each observation kernel performs 3 kinds of readout, all of which are concatenated to create kernel_number*3 total features.

        Where A is an array of agents inside a kernels width, for a given kernel it's readouts are:

        1. A positions summed
        2. positions of A multiplied by the _x_ velocities of A, summed
        3. positions of A multiplied by the _y_ velocities of A, summed

        For specifics and equations, see Section II.C, Lymburn et al (2021).

        Parameters
        ----------
        replica : self.replica#
            Class instance expects to be passed one of its own attibutes, either replica1 or replica2.

        Returns
        -------
        np.ndarray or np.memmap
            matrix of shape (N,F) where N is the number of time steps and F is the number of features, but specifically in this case F = kernel_number*3.
        """

        positions = replica['positions']
        velocities = replica['velocities']
        
        time_steps = positions.shape[0]
        kernels = self.kernel_number
        features = kernels*3 #x3 because 3 readouts occur as specified in the paper

        centers = self.centers
        widths = self.widths
        c_sq = np.sum(centers**2, axis=1)
        
        if self.memory_map:
            npy, npy_path = self._create_mmap('kernel_readout_',(time_steps,features))
            self.tmp_paths.append(npy_path)
        
        chunk_starts = list(range(0, time_steps, self.chunk_size))

        r1_t = []
        r2_t = []
        r3_t = []
        for start in tqdm(chunk_starts, desc="Serial Chunks"):
            chunk_end = min(start + self.chunk_size, time_steps)

            #of this chunk
            pos_c = positions[start:chunk_end]
            velx_c = velocities[start:chunk_end,:,0,None]
            vely_c = velocities[start:chunk_end,:,1,None]

            pos_sq = np.sum(pos_c**2, axis=2, keepdims=True)

            cross = np.dot(pos_c, centers.T)

            #(x-c_m)^2 expand the brackets vvvvv :D
            e_numer = pos_sq + c_sq[None, None, :] - 2.0 * cross
            e_denom = 2 * widths[None, None, :]

            psi = np.exp(-e_numer / e_denom)

            r1 = np.sum(psi, axis=1)
            r2 = np.sum(psi * velx_c, axis=1)
            r3 = np.sum(psi * vely_c, axis=1)

            if self.memory_map:
                #since features are a vector, below indexs the vector ranges 0-199 is r1 readout, 200-399 is r2, 400-600 is r3
                npy[start:chunk_end,0:kernels] = r1
                npy[start:chunk_end,kernels:kernels*2] = r2
                npy[start:chunk_end,kernels*2:kernels*3] = r3
            else:
                r1_t.append(r1)
                r2_t.append(r2)
                r3_t.append(r3)

        if self.memory_map:
            npy.flush()
            return npy
        else:
            r1_t = np.vstack(r1)
            r2_t = np.vstack(r2)
            r3_t = np.vstack(r3)
            return np.concatenate([r1_t, r2_t, r3_t], axis=1)


    def v1(self,sv1,sv2):
        sv1 = sv1 - np.mean(sv1,axis=0)
        sv2 = sv2 - np.mean(sv2,axis=0)

        """ condition explaination:
                1. take unit circle (sphere/hypersphere)
                2. apply matrix to it
                3. circle becomes eliptic
                4. sigma_min is the shortest axis and sigma_max is the longest axis

            Condition is the ratio between sigma_max and sigma_min. If the ratio between them is larger than one,
            it suggests that for some (vector x matrix) multiplication the vector might be unevenly strecthed across dimensions.
            Regularily this is fine and in fact informative, although if the condition is greater than 1e8 (10mil) floating point
            errors will occur and so data will be lost.
        """

        #calculating auto-covariance
        cxx = (sv1.T@sv1)/sv1.shape[0]

        # "To ensure numerical stability, we add a small regularization term" - lymburn et al
        cxx_reg = cxx + 1e-10 * np.eye(len(cxx))
        Sigma, Q = np.linalg.eigh(cxx_reg) # eigendecompoise the autocovariance
        assert np.linalg.cond(cxx_reg) < 1e16, \
            f"{np.linalg.cond(cxx_reg)} - condition of matrix exceeds floating point limit 1e16 so data loss will be incurred by any transformation" 

        reconstructed = Q @ np.diag(Sigma) @ Q.T
        assert np.allclose(cxx_reg, reconstructed)

        eig_inv_sqrt = 1/np.sqrt(Sigma)
        Sigma_inv = np.diagflat(eig_inv_sqrt)

        transformation = Q @ Sigma_inv @ Q.T

        sv1o = sv1@transformation
        sv2o = sv2@transformation

        # The covariance of the whitened data MUST be the Identity matrix
        I = (sv1o.T @ sv1o) / sv1o.shape[0]
        assert np.allclose(np.eye(sv1o.shape[1]),I,atol=1e-1), "Normalisation transform did not produce a autocovariance which makes an identity matrix"

        #cross-covariance
        css = (sv1o.T @ sv2o)/sv1o.shape[0]

        '''
        "While this is only true in the limit of infinite trajectories, we can enforce the structure by averaging on the diagonal and off-diagonal elements and thus better approximate the asymptotic behavior."
        
            whilst in the context of the paper this doesn't seem to be in relation to the observation kernels, it is also true that two 
            kernel replicas aren't t->inf so produce an asymetric matrix and so an eigendecomposition doesn't work
        '''
        css_symm = (css + css.T) / 2
        assert np.allclose(css_symm,css_symm.T), "failed to make the matrix symetric"

        # eigendecompoise the autocovariance
        gamma2 = np.linalg.eigvalsh(css_symm)
        consistent_capacity = np.trace(css_symm) # or could sum gamma2

        assert np.allclose(np.sum(gamma2),consistent_capacity)

        return consistent_capacity,gamma2


    def faithful(self,sv1,sv2):
        """
        This is a methodology for calculating the consistency profile. Named for the fact that it's the most literal interpretation of
        the techniques described in section 1 of the appendix in Lymburn et al (2021).

        Parameters
        ----------
        sv1 : np.ndarray, shape(N,F)
            Vectorised reservoir state of replica1. Assumed to have been generated using `get_reservoir_state_vectorised()`. 
            Numpy array with shape N time_steps/samples and F features/modes.
        sv2 : np.ndarray, shape(N,F)
            Same as sv1 except for replica2.

        Returns
        -------
        consistent_capacity : float
            
            - theta 
            - consistent capacity
            - trace of the cross-covariance of `sv1` and `sv2`

        gamma_sqaured : list[floats]

            - eigenvalues of the cross-covariance of `sv1` and `sv2`

        Notes
        -----
        This function does not demonstrate the most efficient way of performing these calculations, rather, it's intended to be very verbose to make it more comprehendible.
        """        
        
        '''
            "responses may be labeled x(t) and x′(t) and are assumed to have zero mean"
                this is why I center when calculating the covariance
        '''
        x1 = self._center(sv1)
        x2 = self._center(sv2)

        '''
            "First, the covariance matrix is calculated as 
                [Cxx]ij =〈xi(t) xj(t)"

            center seperatly here for optimisation reasons
        '''

        Cxx = self._covariance(sv1,center=True)

        '''
            "To ensure numerical stability, we add a small regularization term 10−9 × I to the covariance matrix prior to calculating T◦."
        '''
        Cxx_reg = Cxx + 1e-9 * np.eye(len(Cxx))


        '''
            "Eigendecomposition of this positive semi-definite matrix reads 
                Cxx = QΣ²Qᵀ "
        '''
        Sigma2, Q = np.linalg.eigh(Cxx_reg)
        sigma_inverse = np.diag(1/np.sqrt(Sigma2))


        '''
            "The reservoir states are normalized with the transformation 
                T◦ = QΣ⁻¹Qᵀ "
        '''
        To = Q @ sigma_inverse @ Q.T


        '''
            "In the new coordinates x◦(t) = T◦x(t)"

                Cosmo note: this is where the paper begins to be unclear. 
                            It's suggested that the transform should be applied
                            to both responses x(t), x'(t) and whilst 
                            the transform should work for x(t) (the first response),
                            it's was calculated from x(t)'s autocovariance, and
                            therefore wouldn't neccesarily be an effective
                            normalisation function for x'(t). This is because the 
                            responses aren't using infinite samples and therefore
                            wont have identical auto-covariance matricies.

        '''
        # swapped the term position to properly match the shapes
        X1o = self._sv_transform(x1,To)
        X2o = self._sv_transform(x2,To)

        '''
            "cross-covariance matrix of the two replicas [Css]ij = 〈s◦,i(t)s◦,j(t)〉 = 〈x◦,i(t)x'◦,j(t)〉."
        '''
        Css = self._covariance(X1o,X2o,center=False)


        '''
            "The eigendecomposition of this positive semi-definite matrix reads Css = Qss Σ²ss Qᵀss>. 
            The diagonal entries of 62 ss are the consistency correlations γ 2 k ."
        '''
        Sigma2_ss = np.linalg.eigvals(Css) #using np.linalg.eigvals because Css isn't symetric


        '''
            "The diagonal entries of Σ²ss are the consistency correlations γ²k ."
                cosmo note: this step below is pointless code wise and simply is used to state that the eigenvalues are the gamma squared features
        '''
        gamma_squared = Sigma2_ss


        '''
            appendix (1) equation (A4), defines that the consistent capacity is the trace of Css
        '''
        consistent_capacity = np.trace(Css)

        return consistent_capacity,gamma_squared

    #TODO get rid of this function and the one in flat readout, instead put the functionality in the super function and check against the object instance when determining whether the methodology is valid. Can even add a message saying like "Methodology not found, maybe you intended to create a KernelReadout object?" etc etc
    def calc_consistency_profile(self,sv1,sv2, methodology:str):
        options = [self.faithful.__name__,self.v1.__name__]

        if methodology not in options:
            raise Exception(f"Available methods for KernelReadout are {options}")
        else:
            methodology = self.__getattribute__(methodology)

        return methodology(sv1,sv2)


class NaiveReadout(ObservationAndPrediction):
    """
        Subclass of `ObservationAndPrediction`. 
        
        _"first considering the naïve approach of taking the two position coordinates 
        of the N agents in the swarm and forming a 2N node reservoir."_ - Lymburn et al

        Only support analysis between two replicas of a simulation.

        See Also
        --------
        `ObservationAndPrediction` : Abstract Base Class for manging, creating readouts, and analysing replicas.
        `KernelReadout` : Subclass which is used to generate observation kernels and perform kernel readouts.
        `COMReadout` : Subclass which is used to perform a center of mass readout.
    """

    def __init__(self,replica1,replica2,washout,chunk_size):
        super().__init__(replica1,replica2,washout,chunk_size)
    

    def get_reservoir_state_vectorised(self, replica):


        x = replica['positions']

        time_steps,num_boids,_ = x.shape
        features = num_boids*2 #x and y positions

        if self.memory_map:
            npy, npy_path = self._create_mmap('flat_readout_',(time_steps,features))
            self.tmp_paths.append(npy_path)

            chunk_starts = range(0, time_steps, self.chunk_size)
            for chunk_start in tqdm(chunk_starts, desc="Flattening Chunks"):
                chunk_end = min(chunk_start + self.chunk_size, time_steps)
                chunk_data = x[chunk_start:chunk_end]
                npy[chunk_start:chunk_end] = chunk_data.reshape(chunk_data.shape[0],chunk_data.shape[1]*chunk_data.shape[2])

            npy.flush()
            return npy

        pos_flattened = x.reshape(x.shape[0],x.shape[1]*x.shape[2]) # flattens the x and y positions into a single vector
        return pos_flattened
    

    def faithful(self, sv1, sv2):
        '''
            implements the equations outlined in the appendix A1-A4
        '''
        time_steps,modes = sv1.shape

        #force them to have zero mean
        x1 = sv1 - np.mean(sv1,axis=0)
        x2 = sv2 - np.mean(sv2,axis=0)

        assert np.allclose(np.mean(x1),np.mean(x2))

        #x1 autocovariance
        cxx = (x1.T@x1)/ time_steps

        # "To ensure numerical stability, we add a small regularization term"
        cxx_reg = cxx + (1e-9 * np.eye(modes))

        #transform for normalising readout
        eigenvalues, eigenvectors = np.linalg.eigh(cxx_reg)
        inverse_sigma = np.diagflat(1/np.sqrt(eigenvalues))

        To = eigenvectors @ inverse_sigma @ eigenvectors.T

        #apply transform
        x1o = x1 @ To
        x2o = x2 @ To

        #calculate cross covariance matrix
        cxx = (x1o.T @ x2o)/time_steps
        print(cxx.shape)

        #retrieve eigenvalues (where gamma squared is stated to be the same thing)
        gamma_squared = np.linalg.eigvals(cxx)
        consistent_capacity = np.trace(cxx)
        assert np.allclose(consistent_capacity,np.sum(gamma_squared))

        return consistent_capacity,gamma_squared
    

    def v1(self,sv1,sv2):
        '''
            this version of the consistency profile calculation is attempting to implement the techniques discussed
            in the second part of the appendix. Specifically, "use[ing] the known structure of the system to improve 
            the efficiency of calculating its consistency profile."
        '''
        pass


    def calc_consistency_profile(self,sv1,sv2, methodology:str):
        options = [self.faithful.__name__,self.v1.__name__]#kinda weird not to just put string myself but this feels more robust against my ability to make typos

        if methodology not in options:
            raise Exception(f"Available methods for KernelReadout are {options}")
        else:
            methodology = self.__getattribute__(methodology)

        return methodology(sv1,sv2)
    

class COMReadout(ObservationAndPrediction):
    """
        Subclass of `ObservationAndPrediction`. 
        
        _"we compare the performance of the full swarm reservoir and one made out of the two CoM coordinates only"_ - Lymburn et al

        Only support analysis between two replicas of a simulation.

        See Also
        --------
        `ObservationAndPrediction` : Abstract Base Class for manging, creating readouts, and analysing replicas.
        `KernelReadout` : Subclass which is used to generate observation kernels and perform kernel readouts.
        `NaiveReadout` : Subclass which is used to perform a naive readout.
    """
    
    def __init__(self,replica1,replica2,washout,chunk_size):
        super().__init__(replica1,replica2,washout,chunk_size)
    
    def get_reservoir_state_vectorised(self, replica):
        x = replica['positions']

        time_steps,num_boids,_ = x.shape
        features = num_boids*2 #x and y positions

        if self.memory_map:
            npy, npy_path = self._create_mmap('flat_readout_',(time_steps,features))
            self.tmp_paths.append(npy_path)

            chunk_starts = range(0, time_steps, self.chunk_size)
            for chunk_start in tqdm(chunk_starts, desc="Flattening Chunks"):
                chunk_end = min(chunk_start + self.chunk_size, time_steps)


                chunk_data = x[chunk_start:chunk_end]
                npy[chunk_start:chunk_end] = np.mean(chunk_data,axis=1)

            npy.flush()
            return npy
        
        
        pos_flattened = np.mean(x,axis=1)

        #pos_normalised = (pos_flattened - np.mean(pos_flattened,axis=0))/np.std(pos_flattened, axis=0)
        #return pos_normalised
        return pos_flattened
    
    def faithful(self, sv1, sv2):
        '''
            implements the equations outlined in the appendix A1-A4
        '''
        time_steps,modes = sv1.shape

        #force them to have zero mean
        x1 = sv1 - np.mean(sv1,axis=0)
        x2 = sv2 - np.mean(sv2,axis=0)

        assert np.allclose(np.mean(x1),np.mean(x2))

        #x1 autocovariance
        cxx = (x1.T@x1)/ time_steps

        # "To ensure numerical stability, we add a small regularization term"
        cxx_reg = cxx + (1e-9 * np.eye(modes))

        #transform for normalising readout
        eigenvalues, eigenvectors = np.linalg.eigh(cxx_reg)
        inverse_sigma = np.diagflat(1/np.sqrt(eigenvalues))

        To = eigenvectors @ inverse_sigma @ eigenvectors.T

        #apply transform
        x1o = x1 @ To
        x2o = x2 @ To

        #calculate cross covariance matrix
        cxx = (x1o.T @ x2o)/time_steps
        print(cxx.shape)

        #retrieve eigenvalues (where gamma squared is stated to be the same thing)
        gamma_squared = np.linalg.eigvals(cxx)
        consistent_capacity = np.trace(cxx)
        assert np.allclose(consistent_capacity,np.sum(gamma_squared))

        return consistent_capacity,gamma_squared
    
    def calc_consistency_profile(self,sv1,sv2, methodology:str):
        options = [self.faithful.__name__]#kinda weird not to just put string myself but this feels more robust against my ability to make typos

        if methodology not in options:
            raise Exception(f"Available methods for KernelReadout are {options}")
        else:
            methodology = self.__getattribute__(methodology)

        return methodology(sv1,sv2)