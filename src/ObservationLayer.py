import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import Ridge
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

    def __init__(self,replica1,replica2,washout,chunk_size=5000,cleanup_tmps=True):
        self.replica1 = replica1
        self.replica2 = replica2
        self.washout_data(washout)
        self.lorenz = replica1.get('predator_positions')
        


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
        '''
            - Find a list of tempory files.
            - Goes through live instances of the ObservationAndPrediction objects, and subtracts any tempfile references from the aformentioned list.
            - Deletes the remaining tmp files in the list.
        '''
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
        def wash(data):
            data['positions'] = data.get('positions')[washout:]
            data['velocities'] = data.get('velocities')[washout:]
            data['predator_positions']= data.get('predator_positions')[washout:]
            data['time_steps']= data.get('time_steps') - washout
            return data
        
        self.replica1 = wash(self.replica1)
        self.replica2 = wash(self.replica2)

    @abstractmethod
    def get_reservoir_state_vectorised(self,data,memmap=None):
        pass

    @abstractmethod
    def calc_consistency_profile(self,sv1,sv2,method):
        pass

    @staticmethod
    def _create_mmap(prefix, shape, dtype='float64'):
        '''
            returns:
                - np.memmap object
                - path to temporary file created
        '''
        os.makedirs(TMP_PATH, exist_ok=True)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix='.npy', dir=TMP_PATH)
        path = tmp_file.name
        tmp_file.close()
        print(f"\nMade tmp file at {path}")
        return np.memmap(path, dtype=dtype, mode='w+', shape=shape), path

    def _sv_mean(self,sv):
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

    def _covariance(self,sv1,sv2=None,center=False):
        '''
            Parameters:
                sv1: 
                    - state vector with shape (T,F) T samples/timesteps and F features/modes
                sv2: 
                    - state vector with shape (T,F) T samples/timesteps and F features/modes
                    - if sv2 is None, autocovariance is calculated with sv1
                center:
                    - True: centers the state vectors using the values found by internal function _sv_mean
                    - False: assumes the state vectors already have 0 mean


            Related methods:
                - _chunked_covariance 
        '''

        if sv2 is None:
            sv2 = sv1
            if center:
                sv1_mean = self._sv_mean(sv1)
                sv2_mean = sv1_mean
        else:
            if center:
                sv1_mean = self._sv_mean(sv1)
                sv2_mean = self._sv_mean(sv2)
        if not center:
            sv1_mean = np.zeros(len(sv1),dtype='float64')
            sv2_mean = np.zeros(len(sv2),dtype='float64')


        if self.memory_map:
            time_steps,feats1 = sv1.shape
            _, feats2 = sv2.shape

            covariances = np.zeros((feats1, feats2), dtype='float64')

            for chunk_start in trange(0,time_steps,self.chunk_size):
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

    def _get_lorenz_targets(self,futures,start=0,end=None):
        '''
            Params:
                futures:
                    how many time steps into the future is being attempted to be predicted
        '''
        lorenz_x = self.lorenz[start:end,0]
        return lorenz_x[futures:]

    def ridge_prediction(self,state_vector,ridge_beta=0.1,traintest_split=0.6,prediction_distance=1):
        """
            Parameters:
                replica (np.ndarray): one of the replica objects inside the obj either obj.replica1 or obj.replica2 
                ridge_beta (float): alpha 
                traintest_split (float): fraction of data used for testing
                prediction_distance (int): time steps into the future you want the replica to attempt to predict

            Returns:
                prediction (np.ndarray): array shape (N) where N is time_steps and each index is the predicted lorenz x positions at said timestep.
                corr_coef (float): the correlation coefficiant found against the prediction and the test split.
        """
        split_idx = int(len(state_vector)*traintest_split)
        state_training = state_vector[:split_idx]
        state_testing = state_vector[split_idx:]

        x_train = state_training[:-prediction_distance]
        y_train = self._get_lorenz_targets(prediction_distance,end=split_idx)
        print(x_train.shape)
        print(y_train.shape)

        x_test = state_testing[:-prediction_distance]
        y_test = self._get_lorenz_targets(prediction_distance,start=split_idx)

        ridge = Ridge(alpha=ridge_beta,fit_intercept=False)

        ridge.fit(x_train,y_train)

        prediction = ridge.predict(x_test)

        corr_coef = np.corrcoef(y_test,prediction)[0,1]
        return prediction, corr_coef

    def plot_ridge_prediction(self,prediction,corr_coef,x_range,pop_out=False):


        lorenz_x = self.lorenz[:,0]

        fig, ax = plt.subplots(figsize=(20, 6))
        plt.subplots_adjust(bottom=0.2)

        window_size = 2000
        max_index = len(lorenz_x)

        ax.plot(lorenz_x, color='red', label='lorenz_x')
        ax.plot(prediction, color='blue', linestyle='dashed', label='Prediction')
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.xlabel('time')
        plt.ylabel('lorenz_x(t+1)')
        plt.title(f'correlation coeficiant R: {corr_coef}')

        if pop_out:
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
        else:
            ax.set_xlim(x_range)

        return ax
    
    def _calc_norm_transform(self,Q,eigen_values):
        """
            Parameters:
                Q:
                    the eigenvector matrix of the given covariance matrix
                eigen_values:
                    the eigenvalue matrix of the given covariance matrix
            Returns:
                the normalisation transform = QΣ⁻¹Qᵀ
        """
        eig_inv_sqrt = 1/np.sqrt(eigen_values)
        Sigma_inv = np.diagflat(eig_inv_sqrt)
        return Q @ Sigma_inv @ Q.T

    def plot_consistency_profile(self,consistent_capacity=None,gamma2_vector=None,truncated_to=100,show_consistent_capacity=True,dpi=200):
        if consistent_capacity is None or gamma2_vector is None:
            consistent_capacity = self.consistent_capacity
            gamma2_vector = self.gamma2_vector

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
            middle_text = r'$\Theta$' + f'={np.round(consistent_capacity,decimals=1)}'
            ax.text(ax.get_xbound()[1]/2, ax.get_ybound()[1]/2, middle_text, fontweight='bold', horizontalalignment='center')   

        return ax


class KernelReadout(ObservationAndPrediction):
    def __init__(self,replica1,replica2,kernel_number,washout,chunk_size):
        super().__init__(replica1,replica2,washout,chunk_size)
        self.kernel_number = kernel_number
        self.centers, self.widths = self.generate_kernels()

    def generate_kernels(self):
        '''
            A kernel has a center and a width. 
            The position c_m is found by getting a random position of a boid at a random time.
            The width w_m is the distance between said random boid and it's 5th closest neighber
        '''

        positions = self.replica2['positions']
        time_steps = self.replica2['time_steps']
        boid_count = self.replica2['boid_count']
        
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
    
    def get_reservoir_state_vectorised(self, run):
        positions = run['positions']
        velocities = run['velocities']
        
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
        '''
            this version of the consistency profile attempts to expand on the methods as exactly described in lymburn.
            This is because the entirely faithful implementation doesn't produce similar consistancy profiles
        '''
        
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

        transformation = self._calc_norm_transform(Q, Sigma)

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
        '''
            this is the most faithful implementaiton of what was described in the paper

            sv1 and sv2 have the shape (N,F) N samples, F features and are the readouts of the observation kernels

            The code below can be simplified using other np feautures but being verbose for readability and proof of implementation
        '''
        
        '''
            "responses may be labeled x(t) and x′(t) and are assumed to have zero mean"
        '''
        x1 = sv1 - np.mean(sv1,axis=0)
        x2 = sv2 - np.mean(sv2,axis=0)


        '''
            "First, the covariance matrix is calculated as 
                [Cxx]ij =〈xi(t) xj(t)"
        '''
        time_steps = x1.shape[0]-1 #number of samples
        Cxx = (x1.T@x1) / time_steps

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
        X1o = x1 @ To
        X2o = x2 @ To

        '''
            "cross-covariance matrix of the two replicas [Css]ij = 〈s◦,i(t)s◦,j(t)〉 = 〈x◦,i(t)x'◦,j(t)〉."
        '''
        Css = (X1o.T @ X2o) /  X2o.shape[0]


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

    def calc_consistency_profile(self,sv1,sv2, method:str):
        options = [self.faithful.__name__,self.v1.__name__]

        if method not in options:
            raise Exception(f"Available methods for KernelReadout are {options}")
        else:
            method = self.__getattribute__(method)

        return method(sv1,sv2)


class FlatReadout(ObservationAndPrediction):
    
    def __init__(self,replica1,replica2,washout,chunk_size):
        super().__init__(replica1,replica2,washout,chunk_size)
    
    def get_reservoir_state_vectorised(self, data):
        x = data['positions']

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

    def calc_consistency_profile(self,sv1,sv2, method:str):
        options = [self.faithful.__name__,self.v1.__name__]#kinda weird not to just put string myself but this feels more robust against my ability to make typos

        if method not in options:
            raise Exception(f"Available methods for KernelReadout are {options}")
        else:
            method = self.__getattribute__(method)

        return method(sv1,sv2)