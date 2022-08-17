import numpy as np
import time
import pickle
from scipy import sparse

class Network:
    def __init__(self, model_params, train_params, seed = 0):
        # initialize properties of the network
        # very slightly adapted from Elham's initialize_net method
        
        # Set parameters
        N = model_params['N']
        rng = np.random.RandomState(seed)
        std = 1/np.sqrt(model_params['pg']*N)
        
        # initialize weighting matrices
        Pw = np.eye(N)/train_params['alpha_w']
        Pd = np.eye(N)/train_params['alpha_d']
        J = std * sparse.random(N, N, density=model_params['pg'], random_state = seed, data_rvs=rng.randn).toarray()
        
        if train_params['init_dist'] == 'Gauss':
            x = 0.1*rng.randn(N,1)
            wf = (1. * rng.randn(N, model_params['d_output'])) / model_params['fb_var']
            wi = (1. * rng.randn(N, model_params['d_input'])) / model_params['input_var']
            wfd = (1. * rng.randn(N, model_params['d_input'])) / model_params['fb_var']
        
        elif train_params['init_dist'] == 'Uniform':
            print('Uniform Initialization')
            x = 0.1 * (2 * rng.rand(N,1) - 1)
            wf = (2 * rng.rand(N, model_params['d_output']) - 1) / model_params['fb_var']
            wi = (2 * rng.rand(N, model_params['d_input']) - 1) / model_params['input_var']
            wfd = (2 * rng.rand(N, model_params['d_input']) - 1) / model_params['fb_var']
            
        else:
            print('Invalid Distribution: Supported distributions are \'Gauss\' and \'Uniform\'')
        
        wo = np.zeros([N, model_params['d_output']])
        wd = np.zeros([N, model_params['d_input']])
        
        JT = model_params['g']*J + np.matmul(wf, wo.T) + np.matmul(wfd, wd.T)
        model_prs = {'JT': JT, 'Pw': Pw, 'Pd': Pd, 'J': J, 'x': x, 'wf': wf, 'wo': wo, 'wfd': wfd, 'wd': wd, 'wi': wi}
        
        self.params = model_params;
        self.params.update(model_prs)
        self.x = x

    
    def memory_trial(self,input):
        # run a single trial of the memory task, for training or testing
        dt = self.params['dt']
        tau = self.params['tau']
        JT = self.params['JT']
        wi = self.params['wi']
        wo = self.params['wo']
        wd = self.params['wd']
        r = np.tanh(self.x)
        
        print(np.shape(wi)[1])
        print(np.shape(input)[0])
        if np.shape(wi)[1] < np.shape(input)[0]:
            dx = -self.x + np.matmul(JT, r) + np.matmul(np.concatenate((np.ones([np.shape(wi)[0],1]),wi),axis=1),input.reshape(-1,1))
        else:
            dx = -self.x + np.matmul(JT, r) + np.matmul(wi, input.reshape(-1,1))
        self.x = self.x + (dx*dt) / tau
        r = np.tanh(self.x)
        z = np.matmul(wo.T, r)
        zd = np.matmul(wd.T, r)
        
        return z, zd
        
    
    def update_weights(self, train_prs, iter, dummy_var, target_var):
        # update the weights of the neural network during training
        update_step = train_prs['update_step']
        dt = self.params['dt']
        Pd = self.params['Pd']
        Pw = self.params['Pw']
        wo = self.params['wo']
        wd = self.params['wd']
        wo_dot, wd_dot = np.zeros([1,]), np.zeros([2,])
        x = self.x
        r = np.tanh(x)
        if 'z' in self.params.keys():
            z = self.params['z']
        else:
            z = np.matmul(wo.T, r)
        zd = np.matmul(wd.T, r)
        
        if np.all(dummy_var != 0.):
            if iter % update_step == 0 and iter >= update_step:
                # update dummy weights
                Pdr = np.matmul(Pd, r)
                num_pd = np.outer(Pdr, np.matmul(r.T, Pd))
                denom_pd = 1 + np.matmul(r.T, Pdr)
                Pd -= num_pd / denom_pd
                self.params['Pd'] = Pd
                
                target_d = np.reshape(dummy_var, [-1, 1])
                ed_ = zd - target_d
                
                Delta_wd = np.outer(Pdr, ed_) / denom_pd
                wd -= Delta_wd
                self.params['wd'] = wd
                
                wd_dot = np.linalg.norm(Delta_wd / (update_step * dt), axis=0, keepdims=True)
            
        if np.any(target_var != 0.):
            if iter % update_step == 0 and iter >= update_step:
                Pr = np.matmul(Pw, r)
                num_pw = np.outer(Pr, np.matmul(r.T, Pw))
                denom_pw = 1 + np.matmul(r.T, Pr)
                Pw -= num_pw / denom_pw
                self.params['Pw'] = Pw
                
                #target = np.reshape(target_var, [self.params['d_output'], 1])
                target = np.reshape(target_var, [1, 1])
                e_ = z - target
                #print('e: ', e_)
                #print('\n')
                
                Delta_w = np.outer(Pr, e_) / denom_pw
                wo -= Delta_w
                self.params['wo'] = wo
                
                wo_dot = np.linalg.norm(Delta_w / (update_step * dt))
        
        self.params['JT'] = self.params['g']*self.params['J'] + np.matmul(self.params['wf'], wo.T) + np.matmul(self.params['wfd'], wd.T)
        return wo_dot, wd_dot
    
    def save_ICs():
        # save the initial conditions leading to a given response in testing
        pass
    
    def remove_neurons(self):
        # remove neurons, i.e., set cells in the JT matrix to 0
        pct_rmv = self.params['pct_rmv']
        x = self.x
        xlen = np.size(x)
        JT = self.params['JT']
        J = self.params['J']
        Pd = self.params['Pd']
        Pw = self.params['Pw']
        wi = self.params['wi']
        wo = self.params['wo']
        wd = self.params['wd']
        wf = self.params['wf']
        wfd = self.params['wfd']
        
        num_to_rmv = round(pct_rmv*xlen)
        rmv_indices = np.random.randint(0,xlen,(num_to_rmv))
        
        for i in range(num_to_rmv):
            x = np.concatenate((x[0:rmv_indices[i]],x[rmv_indices[i]+1:]))
            JT = np.concatenate((JT[0:rmv_indices[i],:],JT[rmv_indices[i]+1:,:]),axis=0)
            JT = np.concatenate((JT[:,0:rmv_indices[i]],JT[:,rmv_indices[i]+1:]),axis=1)
            J = np.concatenate((J[0:rmv_indices[i],:],J[rmv_indices[i]+1:,:]),axis=0)
            J = np.concatenate((J[:,0:rmv_indices[i]],J[:,rmv_indices[i]+1:]),axis=1)
            Pd = np.concatenate((Pd[0:rmv_indices[i],:],Pd[rmv_indices[i]+1:,:]),axis=0)
            Pd = np.concatenate((Pd[:,0:rmv_indices[i]],Pd[:,rmv_indices[i]+1:]),axis=1)
            Pw = np.concatenate((Pw[0:rmv_indices[i],:],Pw[rmv_indices[i]+1:,:]),axis=0)
            Pw = np.concatenate((Pw[:,0:rmv_indices[i]],Pw[:,rmv_indices[i]+1:]),axis=1)
            wi = np.concatenate((wi[0:rmv_indices[i],:],wi[rmv_indices[i]+1:,:]),axis=0)
            wo = np.concatenate((wo[0:rmv_indices[i],:],wo[rmv_indices[i]+1:,:]),axis=0)
            wd = np.concatenate((wd[0:rmv_indices[i],:],wd[rmv_indices[i]+1:,:]),axis=0)
            wf = np.concatenate((wf[0:rmv_indices[i],:],wf[rmv_indices[i]+1:,:]),axis=0)
            wfd = np.concatenate((wfd[0:rmv_indices[i],:],wfd[rmv_indices[i]+1:,:]),axis=0)
            
        self.params['JT'] = JT
        self.params['J'] = J
        self.params['Pd'] = Pd
        self.params['Pw'] = Pw
        self.params['wi'] = wi
        self.params['wo'] = wo
        self.params['wd'] = wd
        self.params['wf'] = wf
        self.params['wfd'] = wfd
        self.params['N'] = np.size(x)
        self.x = x
        #print(str(100*pct_rmv) + '% of neurons removed\n')
        
    def save_network(self, name=None, prefix='train', dir=None):
        file_name = prefix + '_' + name
        print(dir + file_name)
        #f = open(dir + file_name, 'wb')
        with open(dir + file_name, 'wb') as f:
            pickle.dump((self.params,self.x), f, protocol=-1)
        