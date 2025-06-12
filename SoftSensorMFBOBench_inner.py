#!C:\Users\~\Anaconda3\envs\310

"""
Created on Fri Jul 21 12:50:49 2023

@author: Paolo Conti, Johannes Lips

Inner loop for combining ANN Multi-Fidelity Regression with Bayesian Optimization (BO) for Hierarchy Determination in MATLAB
When combined with MATLAB, MATLAB calls this function as part of the BO procedure, the flag MATLAB is automatically set to consider this.
The shebang should be set appropriately at the top of this script.

"""

#%% === MATLAB Setup ===
if 'externalInput' in globals():
    MATLAB = True
else:
    MATLAB = False
    
#%% === Packages ===
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam,Nadam,Adamax
from time import perf_counter

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Add, Lambda, BatchNormalization
from tensorflow.keras.layers import concatenate
import tensorflow as tf
#if not GPU, just comment the following line
#tf.config.list_physical_devices('GPU')

import sys
sys.path.insert(0, './progressive_network')

from progressive_network import MultifidelityNetwork
#%% === Inputs ===
seed = 0

#Choose the benchmark case
case = 'discontinuousNoise'
example = './1D_Benchmark_' + case

scaling = True
train = True
validation = False
save = False
original_order = True

Nhf = 8 #number of points for training
Nhf_total = 14 #total number of points for training and validation
N_test = 100 #number of points for testing

# only relevant when not using MATLAB as outer loop
manual_fidelity_order = [1, 0]

#%% === Model definitions ===

#1D Benchmark: Discontinuous functions
lowfid0 = lambda x: 30*x -10
lowfid1 = lambda x: (0.5*(6.*x-2.)**2*np.sin(12.*x-4) + 10*(x-0.5) -5)*(x<0.5) + (3+0.5*(6.*x-2)**2*np.sin(12.*x-4)+10*(x-0.5)-5)*(x>0.5)
highfid = lambda x: (2*lowfid1(x)- 20*x+20)*(x<0.5) + (4 + 2*lowfid1(x)- 20*x+20)*(x>0.5) 
N_noise = 200
def noise1(x, seed=1):
    # generate a high resolution noise vector that will be interpolated to systematically get the same noise characteristics
    np.random.seed(seed)
    x_test = np.linspace(0,1,N_noise)
    return np.interp(x, x_test, np.random.random(x_test.shape))
def cumNoise1(x, seed = 1):
    # cumulative noise for modelling drift
    np.random.seed(seed)
    x_test = np.linspace(0,1,N_noise)
    return np.interp(x, x_test, np.cumsum(np.random.random(x_test.shape)))

lowfid2 = lambda x: lowfid1(x) + 4*(noise1(x,1)-0.5)
lowfid3 = lambda x: lowfid1(x) + 1.05e-1*cumNoise1(x,2)
lowfid4 = lambda x: lowfid1(x) - 8e-2*cumNoise1(x,3) + 2*(noise1(x,4)-0.5)
lowfid5 = lambda x: 8*(noise1(x,5)-0.5)


#%% === Create dataset ===

#Input
xhf_total = np.linspace(0,1,Nhf_total+1)-0.03
xhf = xhf_total[2::2]
xhf_val = xhf_total[1::2]
xhf_test = np.linspace(0,1,N_test)

if not MATLAB:
    #Plot
    plt.figure()
    plt.plot(xhf_test, lowfid0(xhf_test), 'k--', label = "$f_{LF^{(0)}}$")
    plt.plot(xhf_test, lowfid1(xhf_test), 'r--', label = "$f_{LF^{(1)}}$")
    plt.plot(xhf_test, lowfid2(xhf_test), 'b--', label = "$f_{LF^{(2)}}$")
    plt.plot(xhf_test, lowfid3(xhf_test), 'g--', label = "$f_{LF^{(3)}}$")
    if 'lowfid4' in globals():
        plt.plot(xhf_test, lowfid4(xhf_test), 'm--', label = "$f_{LF^{(4)}}$")
    if 'lowfid5' in globals():
        plt.plot(xhf_test, lowfid5(xhf_test), 'c--', label = "$f_{LF^{(5)}}$")
    plt.plot(xhf_test, highfid(xhf_test), 'r-', label = "$f_{HF}$")
    plt.plot(xhf, highfid(xhf), 'ro')
    plt.plot(xhf_val, highfid(xhf_val), 'kx')
    plt.xlabel('x')
    plt.legend(ncol = 2)
    plt.show()


#%% Define fidelity order:
    
original_fidelity_order = [lowfid0, lowfid1, lowfid2, lowfid3, lowfid4, lowfid5]
original_fidelity_labels = [0, 1, 2, 3, 4, 5]

if not MATLAB:
    # manually set some order 
    fidelity_order_labels = manual_fidelity_order
else:
    # get fidelity order from MATLAB parameter "fidelity_order_labels" and convert to correct format
    fidelity_order_labels = np.array(fidelity_order_labels, dtype = int).flatten()

n_models = len(fidelity_order_labels) #redundant if not MATLAB, else necessary to define variable   
fidelity_order = [original_fidelity_order[i] for i in fidelity_order_labels]
print('Number of models: ', n_models, '\nFidelity order', fidelity_order_labels)

#%% === Scaling ===
if scaling:
    scale = 27.
    scale_param = 1.
    scale_time = 1.
else:
    scale =  1.
    scale_param = 1.
    scale_time = 1.

y = highfid(xhf) / scale
y = y.reshape(-1,1)
y_val = highfid(xhf_val) / scale
y_val = y_val.reshape(-1,1)
y_test = highfid(xhf_test) / scale
y_test = y_test.reshape(-1,1)

#%% === Hyperparameters of multifidelity training ==
n_sim = 5
Nepo = 2500
patience = 100

params = {'lr' :1e-3, 
                'kernel_init' : 'glorot_uniform', 
                'opt' : 'Adam', 
                'activation' : 'tanh',
                'layers_encoder' : [],
                'layers_decoder' : [8, 8, 8], 
                'l2weight' : 1e-4, 
                'Nepo' : Nepo,
                'concatenate' : False}

if validation:
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
else:
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
params['model_type'] = model_type = 'Dense'

input_dim = 1
output_dim = 1
latent_dim = 1

#%% === Train multifidelity model ===
full_input_data = []
full_input_data_val = []
full_input_data_test = []
full_pred_sim = []
full_pred_sim_train = []
full_pred_sim_val = []
full_model_list = []
all_goodness_train = [] # list of all goodness levels reached after adding different fidelity stages.
all_goodness_val = [] # list of all goodness levels reached after adding different fidelity stages.

for n_model in range(n_models):
     print('######### model ' + str(n_model+1) + ' of ' + str(n_models)+ ' #########')

     pred_sim = []
     pred_sim_val = []
     pred_sim_train = []
     model_list = []

     #Preprocess data
     x = fidelity_order[n_model](xhf) / scale
     x = x.reshape(-1,1)

     x_val = fidelity_order[n_model](xhf_val) / scale
     x_val = x_val.reshape(-1,1)

     x_test = fidelity_order[n_model](xhf_test) / scale
     x_test = x_test.reshape(-1,1)

     full_input_data.append(x)
     full_input_data_val.append(x_val)
     full_input_data_test.append(x_test)
    
    
     for i in range(n_sim):
         print('sim ' + str(i+1) + ' of ' + str(n_sim))
         prev_models = [full_model_list[j][i] for j in range(n_model)]
         prev_inputs = [full_input_data[j] for j in range(n_model)]
         model = MultifidelityNetwork(params, input_dim = input_dim, latent_dim = latent_dim, output_dim = output_dim, prev_models = prev_models, prev_inputs = prev_inputs)
         name = example + '/models/model' + str(n_model) + '_traj_' + model_type + str(i) + '_HPO'
        
         if scaling:
                 name = name + '_scaled'   

         if train:
             model.autoencoder.compile(loss='mse',optimizer=params['opt'],metrics=['mse'])
             tf.random.set_seed(seed + i)
             np.random.seed(seed + i)
             input_data = [full_input_data[j] for j in range(n_model+1)]
             input_data_val = [full_input_data_val[j] for j in range(n_model+1)]
             if validation:
                 hist = model.autoencoder.fit(input_data,y,epochs=Nepo,batch_size=Nhf,verbose=0,callbacks=[early_stopping],validation_data = (input_data_val, y_val))
             else:
                 hist = model.autoencoder.fit(input_data,y,epochs=Nepo,batch_size=Nhf,verbose=0,callbacks=[early_stopping])
         else:
             model.load_weights(name)
            
         model_list.append(model)
        
         if save:
             model.save_weights(name)

         #Predict
         input_data_test = [full_input_data_test[j] for j in range(n_model+1)]
         y_pred_test = model.predict(input_data_test)[:,0]
         pred_sim.append(y_pred_test)
         y_pred_train = model.predict(input_data)[:,0]
         pred_sim_train.append(y_pred_train)
         y_pred_val = model.predict(input_data_val)[:,0]
         pred_sim_val.append(y_pred_val)
         #plt.plot(xhf_test, y_pred_test, label = 'HF', color = 'blue',  linewidth = 0.3)

     if not MATLAB:
         mean_sim = np.mean(np.array(pred_sim), axis = 0).flatten()
         std_sim = np.std(np.array(pred_sim), axis = 0).flatten()
         fig = plt.figure(figsize=(6,4))
         plt.plot(xhf_test, y_test * scale, label = 'HF', color = 'red')
         plt.plot(xhf_test, mean_sim* scale, label = 'mean', color = 'blue')
         plt.fill_between(xhf_test, (mean_sim - std_sim)* scale, (mean_sim + std_sim)* scale, color='lightblue', alpha=0.5, label='mean $\pm$ std')
         #plt.plot(xhf,y * scale, 'r*')
         plt.xlabel('x')
         plt.legend()
         plt.show()
         plt.plot(xhf, np.mean(np.array(pred_sim_train), axis = 0).flatten()*scale, 'o')
         plt.plot(xhf_val, np.mean(np.array(pred_sim_val), axis = 0).flatten()*scale, 'o')
    
     # evaluate performance
     full_pred_sim.append(pred_sim)
     full_pred_sim_train.append(pred_sim_train)
     full_pred_sim_val.append(pred_sim_val)
    
     all_goodness_train.append(np.mean(np.sum((y.flatten() - np.array(pred_sim_train))**2, axis = 1))) # goodness of training set (used for MF training)
     all_goodness_val.append(np.mean(np.sum((y_val.flatten() - np.array(pred_sim_val))**2, axis = 1))) # goodness of validation set (used for BO training)
    
     full_model_list.append(model_list)

#%% === Plot result ===
if not MATLAB:
    mean_sim = np.mean(np.array(full_pred_sim[-1]), axis = 0)
    std_sim = np.std(np.array(full_pred_sim[-1]), axis = 0)
    fig = plt.figure(figsize=(6,4))
    plt.plot(xhf_test, y_test * scale, label = 'HF', color = 'red')
    plt.plot(xhf_test, mean_sim * scale, label = 'mean', color = 'blue')
    plt.fill_between(xhf_test, (mean_sim - std_sim) * scale, (mean_sim + std_sim) * scale, color='lightblue', alpha=0.5, label='mean $\pm$ std')
    plt.plot(xhf,y * scale, 'r*')
    plt.plot(xhf_val, np.mean(np.array(pred_sim_val), axis = 0).flatten()*scale, 'o')
    plt.xlabel('x')
    plt.legend()
    plt.show()
# %%
