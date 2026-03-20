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
        self.state_vectors = np.array(replicas) #per replica
        self.covariance_matricies = []# per replica
        self.eigen_vectors = []# per replica
        self.eigen_values = []# per replica
        self.normalisation_transforms = []# per replica
        self.sv_norm = []#state vector normalised

        for s in self.state_vectors:

            s_covariance_matrix = self.__calc_covariance_matrix(s)
            self.covariance_matricies.append(s_covariance_matrix)

            s_eigen_values, s_eigen_vectors =  np.linalg.eigh(s_covariance_matrix)
            sigma_squared = np.diagflat(s_eigen_values)

            self.eigen_vectors.append(s_eigen_vectors)
            self.eigen_values.append(sigma_squared)

            s_norm_trans = self.__calc_norm_transform(s_eigen_vectors, sigma_squared)
            self.normalisation_transforms.append(s_norm_trans)
            
            s_vectors_normalised = s@s_norm_trans
            self.sv_norm.append(s_vectors_normalised)

        self.state_vectors = np.array(self.state_vectors)
        self.covariance_matricies = np.array(self.covariance_matricies)
        self.eigen_vectors = np.array(self.eigen_vectors)
        self.eigen_values = np.array(self.eigen_values)

    def __calc_norm_transform(self,Q,Sigma_squared):
        """
            Parameters:
                eigvecs:
                    the eigenvector matrix of the given covariance matrix
                eigvals:
                    the eigenvalue matrix of the given covariance matrix
            Returns:
                the normalisation transform = QΣ⁻¹Qᵀ
        """
        return Q @ np.linalg.inv(np.sqrt(Sigma_squared)) @ Q.T

    def __calc_covariance_matrix(self,state,assume_zero_mean:bool = True):
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
        if not assume_zero_mean:
            #cxx = < ( x_i(t)-xi_mean) * ( xj(t)-xj_mean ) >
            state_mean = np.mean(state,axis=0) # the mean for each state value [x1_mean, ... xn_mean] (where n is the size of the state vector)
            state_zero_mean = state-state_mean # the state vector where each value has zero mean
            assert np.allclose(np.mean(state_zero_mean,axis=0),0)# ensures ts worked
            state = state_zero_mean
        
        time_steps = state.shape[0]
        state_transpose = state.T #gives (S,T)
        cxx=(state_transpose @ state)/time_steps
        
        cxx+=(10**-9)*np.eye(len(cxx)) # "To ensure numerical stability, we add a small regularization term" - lymburn et al
        return cxx


    def calc_x_covariance_matrix(self,replica_state1,replica_state2):
        '''
            cross covariance matrix
        '''
        time_steps = replica_state1.shape[0]
        css = (replica_state1.T@replica_state2)/time_steps
        return css

    def calc_consistent_capacity(self,replica1,replica2):
        css = self.calc_x_covariance_matrix(replica1,replica2)
        trace = np.trace(css)
        return trace
    
        # below would only work if Css is symetric. Which is only a given if replica1 = replica2.T 
        #eigen_values,_ = np.linalg.eig(css)
        #return np.sum(eigen_values)

    def plot_consistency_profile(self,x_covariance_matrix,truncated_to=100,show_consistent_capacity=True,dpi=200):
        '''
            Parameters:
                x_covariance_matrix:
                    the cross covariance matrix between (assuming) two replicas
                truncated_to:
                    default = 100 (per lymburn et al). Limits the width of the graph showing only top 100 covaried feautures.
                show_consistent_capacity:
                    whether the plot should include the consistent capacity in the middle

            Returns:
                ax
        '''
        gamma2_k = np.diag(x_covariance_matrix)
        gamma2_k_ranked = np.flip(np.sort(gamma2_k,))

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
            trace = np.trace(x_covariance_matrix)
            middle_text = r'$\Theta$' + f'={trace}'
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