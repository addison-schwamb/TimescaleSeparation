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
    zd_mat = np.zeros([net_params['d_dummy'], total_steps])
    x_mat = np.zeros([net_params['N'], total_steps])
    r_mat = np.zeros([net_params['N'], total_steps])
    wo_dot = np.zeros([total_steps, net_params['d_output']])
    wd_dot = np.zeros([total_steps, net_params['d_dummy']])

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
    fast_z_mat, fast_zd_mat, fast_x_mat, fast_r_mat, fast_wo_dot, fast_wd_dot = zero_fat_mats(fast_net.params, train_prs, task_prs['t_trial'], is_train=True)
    
    slow_net.params['z'] = fast_z
    trial = 0
    
    for i in range(slow_steps):
        slow_z_mat[:, i] = slow_z
        slow_zd_mat[:, i] = slow_zd.reshape(-1)
        slow_x_mat[:, i] = slow_x.reshape(-1)
        slow_r_mat[:, i] = slow_r.reshape(-1)
        
        slow_z, slow_zd = slow_net.memory_trial(np.concatenate((exp_mat[:,i*fast_steps],fast_zd),axis=None))
        slow_wo_dot[i], slow_wd_dot[i,:] = slow_net.update_weights(train_prs, i, dummy_mat[:,i], target_mat[:,i])
        
        for j in range(fast_steps):
            fast_z_mat[:, i*fast_steps + j] = fast_z
            fast_zd_mat[:, i*fast_steps + j] = fast_zd.reshape(-1)
            fast_x_mat[:, i*fast_steps + j] = fast_x.reshape(-1)
            fast_r_mat[:, i*fast_steps + j] = fast_r.reshape(-1)
            
            fast_z, fast_zd = fast_net.memory_trial(np.concatenate((slow_z,slow_zd),axis=None))
            fast_wo_dot[i], fast_wd_dot[i,:] = fast_net.update_weights(train_prs, i*fast_steps + j, dummy_mat[:,i*fast_steps + j], target_mat[:,i*fast_steps + j])
            slow_net.params['z'] = fast_z
    
    toc = time.time()
    print('\n', 'train time = ' , (toc-tic)/60)
    task_prs['counter'] = i*fast_steps+j
    return slow_net, fast_net, task_prs
    
def test(slow_net, fast_net, train_prs, task_prs, exp_mat, target_mat, input_digits):
    slow_params = slow_net.params
    fast_params = fast_net.params
    slow_steps = int(train_prs['n_test']* task_prs['t_trial'] / slow_params['dt'])
    fast_steps = int(slow_params['dt'] / fast_params['dt'])
    
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
    
    counter = task_prs['counter']
    exp_mat = exp_mat[:, counter+1:]
    target_mat = target_mat[:, counter+1:]
    output = task_prs['output_encoding']
    test_digits = input_digits[train_prs['n_train']+ train_prs['n_train_ext']:]
    i00, i01, i10, i11 = 0, 0, 0, 0
    
    slow_z_mat, slow_zd_mat, slow_x_mat, slow_r_mat, slow_wo_dot, slow_wd_dot = zero_fat_mats(slow_net.params, train_prs, task_prs['t_trial'], is_train=False)
    fast_z_mat, fast_zd_mat, fast_x_mat, fast_r_mat, fast_wo_dot, fast_wd_dot = zero_fat_mats(fast_net.params, train_prs, task_prs['t_trial'], is_train=False)
    
    correct = 0
    trial = 0
    
    for i in range(slow_steps):
        slow_z_mat[:, i] = slow_z
        slow_zd_mat[:, i] = slow_zd.reshape(-1)
        slow_x_mat[:, i] = slow_x.reshape(-1)
        slow_r_mat[:, i] = slow_r.reshape(-1)
        
        slow_z, slow_zd = slow_net.memory_trial(np.concatenate((exp_mat[:,i*fast_steps],fast_zd),axis=None))
        
        for j in range(fast_steps):
            fast_z_mat[:, i*fast_steps + j] = fast_z
            fast_zd_mat[:, i*fast_steps + j] = fast_zd.reshape(-1)
            fast_x_mat[:, i*fast_steps + j] = fast_x.reshape(-1)
            fast_r_mat[:, i*fast_steps + j] = fast_r.reshape(-1)
            
            fast_z, fast_zd = fast_net.memory_trial(np.concatenate((slow_z,slow_zd),axis=None))
            
            if (i*fast_steps+j) % int((task_prs['t_trial'])/ fast_params['dt'] ) == 0 and i*fast_steps+j != 0:
                if test_digits[trial][1] == (0,0) and i00 == 0:

                    slow_r00 = slow_r_mat[:, i-1][:, np.newaxis]
                    slow_x00 = slow_x_mat[:, i-1][:, np.newaxis]
                    fast_r00 = fast_r_mat[:, (i*fast_steps+j)-1][:, np.newaxis]
                    fast_x00 = fast_x_mat[:, (i*fast_steps+j)-1][:, np.newaxis]
                    i00 = 1

                elif test_digits[trial][1] == (0,1) and i01 == 0:

                    slow_r01 = slow_r_mat[:, i-1][:, np.newaxis]
                    slow_x01 = slow_x_mat[:, i-1][:, np.newaxis]
                    fast_r01 = fast_r_mat[:, (i*fast_steps+j)-1][:, np.newaxis]
                    fast_x01 = fast_x_mat[:, (i*fast_steps+j)-1][:, np.newaxis]
                    i01 = 1

                elif test_digits[trial][1] == (1,0) and i10 == 0:

                    slow_r10 = slow_r_mat[:, i-1][:, np.newaxis]
                    slow_x10 = slow_x_mat[:, i-1][:, np.newaxis]
                    fast_r10 = fast_r_mat[:, (i*fast_steps+j)-1][:, np.newaxis]
                    fast_x10 = fast_x_mat[:, (i*fast_steps+j)-1][:, np.newaxis]
                    i10 = 1

                elif test_digits[trial][1] == (1,1) and i11 == 0:

                    slow_r11 = slow_r_mat[:, i-1][:, np.newaxis]
                    slow_x11 = slow_x_mat[:, i-1][:, np.newaxis]
                    fast_r11 = fast_r_mat[:, (i*fast_steps+j)-1][:, np.newaxis]
                    fast_x11 = fast_x_mat[:, (i*fast_steps+j)-1][:, np.newaxis]
                    i11 = 1
                
                print('test_digits: ',input_digits[trial])
                print('z: ',np.around(2*fast_z)/2.0);
                if np.around(2*fast_z)/2.0 == output[sum(test_digits[trial][1])]:
                    correct += 1
                trial += 1
    
    pct_correct = correct/(train_prs['n_test']-1)
    slow_x_ICs = np.array([slow_x00, slow_x01, slow_x10, slow_x11])
    slow_r_ICs = np.array([slow_r00, slow_r01, slow_r10, slow_r11])
    fast_x_ICs = np.array([fast_x00, fast_x01, fast_x10, fast_x11])
    fast_r_ICs = np.array([fast_r00, fast_r01, fast_r10, fast_r11])

    return  slow_net, fast_net, pct_correct, slow_x_ICs, slow_r_ICs, fast_x_ICs, fast_r_ICs, slow_x_mat, fast_x_mat