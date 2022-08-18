# Pseudocode
# Set params
# Create task
# Create blank slow network
# Create blank fast network
# Train both networks simultaneously
# Test

import argparse
import json
import sys
import os
from SPM_task import *
from Network import *
from train_force import *
from posthoc_tests import *

def set_all_parameters( g, pg, fb_var, dt, input_var,  n_train, encoding, seed, init_dist, activation='tanh', isFORCE = False):
    params = dict()

    slow_net_params = dict()
    slow_net_params['d_input'] = 4
    slow_net_params['d_output'] = 1
    slow_net_params['d_dummy'] = 2
    slow_net_params['tau'] = 1
    slow_net_params['dt'] = dt[0]
    slow_net_params['g'] = g[0]
    slow_net_params['pg'] = pg[0]
    slow_net_params['N'] = 1000
    slow_net_params['fb_var'] = fb_var[0]
    slow_net_params['input_var'] = input_var
    params['slow_network'] = slow_net_params
    
    fast_net_params = dict()
    fast_net_params['d_input'] = 3
    fast_net_params['d_output'] = 1
    fast_net_params['d_dummy'] = 2
    fast_net_params['tau'] = 1
    fast_net_params['dt'] = dt[1]
    fast_net_params['g'] = g[1]
    fast_net_params['pg'] = pg[1]
    fast_net_params['N'] = 1000
    fast_net_params['fb_var'] = fb_var[1]
    fast_net_params['input_var'] = input_var
    params['fast_network'] = fast_net_params

    task_params = dict()
    t_intervals = dict()
    t_intervals['fixate_on'], t_intervals['fixate_off'] = 0, 0
    t_intervals['cue_on'], t_intervals['cue_off'] = 0, 0
    t_intervals['stim_on'], t_intervals['stim_off'] = 10, 5
    t_intervals['delay_task'] = 0
    t_intervals['response'] = 5
    task_params['time_intervals'] = t_intervals
    task_params['t_trial'] = sum(t_intervals.values()) + t_intervals['stim_on'] + t_intervals['stim_off']
    task_params['output_encoding'] = encoding  # how 0, 1, 2 are encoded
    task_params['keep_perms'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    task_params['n_digits'] = 9
    task_params['counter'] = 0
    params['task'] = task_params

    train_params = dict()
    train_params['update_step'] = 2  # update steps of FORCE
    train_params['alpha_w'] = 1.
    train_params['alpha_d'] = 1.
    train_params['n_train'] = n_train  # training steps
    train_params['n_train_ext'] = 0
    train_params['n_test'] = 20   # test steps
    train_params['init_dist'] = init_dist
    train_params['activation'] = activation
    train_params['FORCE'] = isFORCE
    train_params['epsilon'] = [0.005, 0.01, 0.05, 0.1]
    params['train'] = train_params
    

    other_params = dict()
    other_params['name'] = str(g) + '_' + str(pg) + '_' + str(fb_var) + '_' + str(seed) + '_' + str(n_train) + '_training_steps'
    print('name is = ',other_params['name']  )
    #str(task_params['output_encoding']) + '_g' + str(net_params['g']) + '_' +
    #  str(train_params['n_train']+ train_params['n_train_ext'])+ 'Gauss_S' + 'FORCE'
    other_params['seed'] = seed  #default is 0
    params['msc'] = other_params

    return params
    
def get_digits_reps():
    with open('allDigCNNMNIST', 'rb') as f:
        z_mean, z_log_var, z_sample = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)

    y_test, x_test = np.array(y_test), x_test.reshape([x_test.shape[0], 28, 28])

    return y_test, z_sample
    
def main(d):
    kwargs = json.loads(d)

    params = set_all_parameters(**kwargs)
    labels, digits_rep = get_digits_reps()
    task_prs = params['task']
    train_prs = params['train']
    slow_net_prs = params['slow_network']
    fast_net_prs = params['fast_network']
    msc_prs = params['msc']
    
    task = sum_task_experiment(task_prs['n_digits'], train_prs['n_train'], train_prs['n_train_ext'], train_prs['n_test'], task_prs['time_intervals'],
                               fast_net_prs['dt'], task_prs['output_encoding'], task_prs['keep_perms'] , digits_rep, labels, msc_prs['seed'])
    exp_mat, target_mat, dummy_mat, input_digits, output_digits = task.experiment()
    
    slow_net = Network(slow_net_prs, train_prs, msc_prs['seed'])
    fast_net = Network(fast_net_prs, train_prs, msc_prs['seed'])
    
    slow_net, fast_net, task_prs = train_time_sep(slow_net, fast_net, train_prs, task_prs, exp_mat, dummy_mat, target_mat, input_digits)
    slow_net, fast_net, pct_correct, slow_x_ICs, slow_r_ICs, fast_x_ICs, fast_r_ICs, slow_x_mat, fast_x_mat = test(slow_net, fast_net, train_prs, task_prs, exp_mat, target_mat, input_digits)
    
    print('Percent Correct: ', str(pct_correct*100), '%')



if __name__ == "__main__":
    print(sys.argv[2])
    main(sys.argv[2])