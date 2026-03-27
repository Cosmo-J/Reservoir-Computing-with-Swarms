import numpy as np
from matplotlib import pyplot as plt
from .SimSaverLoader import SimSaverLoader
from .ObservationLayer import FlatReadout

class PropertyAnalysis:
    '''
    Used to analysise the properties of a reservoir
    '''

    def __init__(self,replicas):
        '''
            readouts should be a list of identical (shape) state vectors. If only one replica, should be passed nested in a list.
            Called readout because in the context of the kernel observation layer readout makes more sense
        '''
        self.state_vectors = [] #per replica
        self.autocovariances = []
        self.normalisation_transforms = []# per replica

        for s in replicas:
            #The paper assumes 0 mean given unlimited time steps so that variance can be calculated correctlty
            #hence below which makes it so that s has a mean 0
            s = s - np.mean(s,axis=0,keepdims=True)
            self.state_vectors.append(s)
            s_autocovariance_matrix = self.__calc_autocovariance_matrix(s)
            self.autocovariances.append(s_autocovariance_matrix)
            s_eigen_values, s_eigen_vectors =  np.linalg.eigh(s_autocovariance_matrix)

            assert np.allclose(s_eigen_vectors@np.diagflat(s_eigen_values)@s_eigen_vectors.T, s_autocovariance_matrix) # validates the eigendecomposition

            s_norm_trans = self.__calc_norm_transform(s_eigen_vectors, s_eigen_values)
            self.normalisation_transforms.append(s_norm_trans)
            
        self.state_vectors = np.array(self.state_vectors)

    def __calc_norm_transform(self,Q,eigen_values):
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

    def __calc_autocovariance_matrix(self,state_vector):
        '''
        Maths:
            Covariance equation:
                    $c_{xx} = \langle x_i(t) * x_j(t)\rangle$   latex version
                    c[i,j] = <xi(t) * xj(t)>                    readable version

                    Makes the assumption that xi averaged over time = 0. In other words, that xs default value is 0, and therefore, that its scale is a measure of its deviation from its norm. And therfore, the covariance, which is multiplying these two together, a combined scale, speaks to their shared deviation from the norm.

                    this is fucking varience 𝜎^2= Σ (x-x_mean)/N 
        Params:
            state:
                assumed to be shape (T,S) where T is the time steps, and s is a concatinated state space. for example, the number of boids in S would be S/4 because S should contain x and y, positions and velocities.
            
            assume_zero_mean:
                if True: the function assumes that a given state vector value has zero mean. 
                If False: the function calculates the mean for each state vector value and subtracts this avg vector from the original state vector ensuring that the state vector does in fact have zero mean for each value

        Returns:
            out:
                the symetric matrix of state S^T S averaged over time
        '''
        
        time_steps = state_vector.shape[0]
        cxx=(state_vector.T @ state_vector)/time_steps
        
        cxx+=(10**-9)*np.eye(len(cxx)) # "To ensure numerical stability, we add a small regularization term" - lymburn et al
        return cxx


    def calc_consistency_profile(self,replica_idx:tuple):
        '''
            cross covariance matrix

            Params:
                replica_idx:
                    a tuple (id1,id2) which will be used to index the given replicas

            Returns:
                Css = Qss Sigma^2ss Q.Tss 
                out:
                    covariance matrix (Css), eigenvector matrix (Qss), eigenvalue matrix (sigma squared), consistent capacity, gamma2_vector
        '''
        
        id1=replica_idx[0]
        id2=replica_idx[1]

        sv1=self.state_vectors[id1]
        sv2=self.state_vectors[id2]

        # average the autocovariance of both the vectors which again is to maintain the assumption that over infinite time the varience between two replicas would be identical for each mode
        average_autocovariance = (self.autocovariances[id1] + self.autocovariances[id1]) / 2
        eigen_values,eigen_vectors = np.linalg.eigh(average_autocovariance)
        transformation = self.__calc_norm_transform(eigen_vectors,eigen_values)

        replica1_norm = sv1 @ transformation
        replica2_norm = sv2 @ transformation

        time_steps = replica1_norm.shape[0]
        css = ( replica1_norm.T @ replica2_norm ) / time_steps

        eigen_values,_ = np.linalg.eigh(css)

        sigma_squared = np.diagflat(eigen_values)
        
        consistent_capacity = np.round(np.trace(sigma_squared),decimals=1) # diagonal entries summed
        gamma2_vector = np.diag(sigma_squared) # diagonal entries in a list


        return consistent_capacity, gamma2_vector


    def plot_consistency_profile(self,replica_idx:tuple,truncated_to=100,show_consistent_capacity=True,dpi=200):
        '''
            Parameters:
                replica_idx:
                    two replica idexes of replicas that where given on the object initialisation. These indexed replicas will be used to perform
                    the cross covariance (between them), the 
                truncated_to:
                    default = 100 (per lymburn et al). Limits the width of the graph showing only top 100 covaried feautures.
                show_consistent_capacity:
                    whether the plot should include the consistent capacity in the middle

            Returns:
                ax
        '''
        consistent_capacity, gamma2_vector = self.calc_consistency_profile(replica_idx) 


        gamma2_k_ranked = np.flip(np.sort(gamma2_vector,))

        plt.rcParams['text.usetex'] = True
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
            middle_text = r'$\Theta$' + f'={consistent_capacity}'
            ax.text(ax.get_xbound()[1]/2, ax.get_ybound()[1]/2, middle_text, fontweight='bold', horizontalalignment='center')   

        return ax


""" run_path = '/Users/cosmo/Dropbox/University/Year 3/Individual Project/code/tests/super_long'
datas = SimSaverLoader.find_npzs(run_path)
fr = FlatReadout(datas[0],datas[1])

v1= fr.get_reservoir_state_vectorised(datas[0])
v2= fr.get_reservoir_state_vectorised(datas[1])

pa = PropertyAnalysis([v1,v2])

ccx = pa.calc_x_covariance_matrix(pa.sv_norm[0],pa.sv_norm[1])
ax = pa.plot_consistency_profile(ccx)
plt.show() """