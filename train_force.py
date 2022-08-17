print('train_force is executing\n')
import time
import pickle
import numpy as np
from scipy import sparse
from drawnow import *
from SPM_task import *

def zero_fat_mats(net_params, train_params, t_trial, is_train=True):

    '''
    initialize zero matrix
    '''

    if is_train:
        total_size = train_params['n_train'] + train_params['n_train_ext']
    elif not is_train:
        total_size = train_params['n_test']

    total_steps = int(total_size * t_trial / net_params['dt'])
    z_mat = np.zeros([net_params['d_output'],total_steps])
    zd_mat = np.zeros([net_params['d_input'], total_steps])
    x_mat = np.zeros([net_params['N'], total_steps])
    r_mat = np.zeros([net_params['N'], total_steps])
    wo_dot = np.zeros([total_steps, net_params['d_output']])
    wd_dot = np.zeros([total_steps, net_params['d_input']])

    return z_mat, zd_mat, x_mat, r_mat, wo_dot, wd_dot
    
def train_time_sep(slow_net, fast_net, train_prs, task_prs, exp_mat, dummy_mat, target_mat, input_digits):
    tic = time.time()
    
    slow_params = slow_net.params
    fast_params = fast_net.params
    slow_steps = int((train_prs['n_train'] + train_prs['n_train_ext'])* task_prs['t_trial'] / slow_params['dt'])
    fast_steps = int(slow_params['dt'] / fast_params['dt'])
    time_steps = np.arange(0, fast_steps, 1)
    
    slow_wo, slow_wd = slow_params['wo'], slow_params['wd']
    fast_wo, fast_wd = fast_params['wo'], fast_params['wd']
    slow_x = slow_net.x
    fast_x = fast_net.x
    slow_r = np.tanh(slow_x)
    fast_r = np.tanh(fast_x)
    slow_z = np.matmul(slow_wo.T, slow_r)
    fast_z = np.matmul(fast_wo.T, fast_r)
    slow_zd = np.matmul(slow_wd.T, slow_r)
    fast_zd = np.matmul(fast_wd.T, fast_r)
    
    slow_z_mat, slow_zd_mat, slow_x_mat, slow_r_mat, slow_wo_dot, slow_wd_dot = zero_fat_mats(slow_net.params, train_prs, task_prs['t_trial'], is_train=True)
    slow_wd_dot = slow_wd_dot[:, 0:slow_params['d_input']]
    fast_z_mat, fast_zd_mat, fast_x_mat, fast_r_mat, fast_wo_dot, fast_wd_dot = zero_fat_mats(fast_net.params, train_prs, task_prs['t_trial'], is_train=True)
    
    slow_net.params['z'] = fast_z
    trial = 0
    
    for i in range(slow_steps):
        slow_z_mat[:, i] = slow_z
        slow_zd_mat[:, i] = slow_zd.reshape(-1)
        slow_x_mat[:, i] = slow_x.reshape(-1)
        slow_r_mat[:, i] = slow_r.reshape(-1)
        
        print(np.shape(np.concatenate((exp_mat[:,i*fast_steps],fast_zd),axis=None)))
        slow_z, slow_zd = slow_net.memory_trial(np.concatenate((exp_mat[:,i*fast_steps],fast_zd),axis=None))
        slow_wo_dot[i], slow_wd_dot[i,:] = slow_net.update_weights(train_prs, i, dummy_mat[:,i], target_mat[:,i])
        
        for j in range(fast_steps):
            fast_z_mat[:, i*fast_steps + j] = fast_z
            fast_zd_mat[:, i*fast_steps + j] = fast_zd.reshape(-1)
            fast_x_mat[:, i*fast_steps + j] = fast_x.reshape(-1)
            fast_r_mat[:, i*fast_steps + j] = fast_r.reshape(-1)
            
            fast_z, fast_zd = fast_net.memory_trial(np.concatenate((slow_z,slow_zd),axis=None))
            fast_wo_dot[i], fast_wd_dot[i,:] = fast_net.update_weights(i*fast_steps + j, dummy_mat[:,i*fast_steps + j], target_mat[:,i*fast_steps + j])
            slow_net.params['z'] = fast_z
            
            if i % int((task_prs['t_trial'])/ params['dt'] ) == 0 and i != 0:
                print('test_digits: ',input_digits[trial])
                print('z: ',np.around(2*z)/2.0);
                trial += 1
    
    toc = time.time()
    task_prs['counter'] = i
    return slow_net, fast_net, task_prs