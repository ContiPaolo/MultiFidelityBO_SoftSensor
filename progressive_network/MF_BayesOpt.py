
#%%
"""
Created on Fri Jul 21 12:50:49 2023

@author: paolo
"""

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
# %%

#######################     CONFIGURATIONS     ##########################
seed = 0

#Choose the benchmark case
case = 'discontinuous'#'discontinuous' #'linear'
example = './1D_Benchmark_' + case

scaling = True
train = True
save = False
original_order = True

if case == 'linear':
    Nhf = 5
elif case == 'discontinuous':
    Nhf = 8
N_test = 100


#%% Model definitions

#1D Benchmark: Linear correlation
if case == 'linear':
    lowfid0 = lambda x: 30*x -10
    lowfid1 = lambda x: np.sin(12*x-4)
    lowfid2 = lambda x: (6*x -2.)**2 * lowfid1(x)
    lowfid3 = lambda x: np.cos(7*x) + 3*np.sin(30*x)
    highfid = lambda x: 0.5*lowfid2(x) + 10*(x-0.5) + 5. 

if case == 'discontinuous':
    #1D Benchmark: Discontinuous functions
    lowfid0 = lambda x: 30*x -10
    lowfid1 = lambda x: ((6.*x-2.)*np.sin(12.*x-4))*(x<0.5) + (3+0.5*(6.*x-2)*np.sin(12.*x-4))*(x>0.5)
    lowfid2 = lambda x: (0.5*(6.*x-2.)**2*np.sin(12.*x-4) + 10*(x-0.5) -5)*(x<0.5) + (3+0.5*(6.*x-2)**2*np.sin(12.*x-4)+10*(x-0.5)-5)*(x>0.5)
    lowfid3 = lambda x: np.cos(7*x) + 3*np.sin(30*x)
    highfid = lambda x: (2*lowfid2(x)- 20*x+20)*(x<0.5) + (4+2*lowfid2(x)- 20*x+20)*(x>0.5) 



#%% Create dataset 

#Input
xhf = np.linspace(0,1,Nhf)
x_test = np.linspace(0,1,N_test)

#Plot
plt.figure()
plt.plot(x_test, lowfid0(x_test), 'k--', label = "$f_{LF^{(0)}}$")
plt.plot(x_test, lowfid1(x_test), 'g--', label = "$f_{LF^{(1)}}$")
plt.plot(x_test, lowfid2(x_test), 'b--', label = "$f_{LF^{(2)}}$")
plt.plot(x_test, lowfid3(x_test), 'r--', label = "$f_{LF^{(3)}}$")
plt.plot(x_test, highfid(x_test), 'r-', label = "$f_{HF}$")
plt.xlabel('x')
plt.legend(ncol = 2)
plt.show()


#%% Define fidelity order:
    
original_fidelity_order = [lowfid0, lowfid1, lowfid2, lowfid3]
original_fidelity_labels = [0, 1, 2, 3]

#Do a permutation of the labels and sort fidelity orders accordingly
np.random.seed(30)
fidelity_order = np.random.permutation(original_fidelity_labels)
print(fidelity_order)

fidelity_order = [original_fidelity_order[i] for i in fidelity_order]
fidelity_order = original_fidelity_order

#%%
#########################     MODEL 0: PREPROCESS    ##########################
###############################################################################

#Scaling
if scaling:
    if case == 'discontinuous':
        scale = 27.
    elif case == 'linear':
        scale = 18.
    scale_param = 1.
    scale_time = 1.
else:
    scale =  1.
    scale_param = 1.
    scale_time = 1.

#Preprocess data 
x0 = fidelity_order[0](xhf) / scale
x0 = x0.reshape(-1,1)

x0_test = fidelity_order[0](x_test) / scale
x0_test = x0_test.reshape(-1,1)

y = highfid(xhf) / scale
y = y.reshape(-1,1)

y_test = highfid(x_test) / scale
y_test = y_test.reshape(-1,1)

#Hyperparameters
n_sim = 5
Nepo = 2500
patience = 50
#if case == 'disco':
params = {'lr' :1e-3, 
                'kernel_init' : 'glorot_uniform', 
                'opt' : 'Adam', 
                'activation' : 'tanh',
                'layers_encoder' : [],
                'layers_decoder' : [8, 8, 8], 
                'l2weight' : 1e-4, 
                'Nepo' : Nepo,
                'concatenate' : False}

pred_sim0 = []
model0_list = []
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)


#%%
#########################     MODEL 0: TRAIN    ##########################
##########################################################################

params['model_type'] = model_type = 'Dense'

input_dim = 1
output_dim = 1
latent_dim = 1

#fig = plt.figure(figsize=(6,6))

for i in range(n_sim):
    print('sim ' + str(i+1) + ' of ' + str(n_sim))
    
    #### Single-fidelity
    with tf.device('/GPU:0'):
        model0 = MultifidelityNetwork(params, input_dim = input_dim, latent_dim = latent_dim, output_dim = output_dim, prev_models = [], prev_inputs = [])
        name = example + '/models/model0_traj_' + model_type + str(i) + '_HPO'
        if scaling:
                name = name + '_scaled'

        if train:
            model0.autoencoder.compile(loss='mse',optimizer=params['opt'],metrics=['mse'])
            tf.random.set_seed(seed + i)
            np.random.seed(seed + i)
            hist0 = model0.autoencoder.fit(x0,y,epochs=Nepo,batch_size=Nhf,verbose=0,callbacks=[early_stopping])
        else:
            model0.load_weights(name)

        model0_list.append(model0)  

        if save:
            model0.save_weights(name)
    
    #Predict
    y_pred0_test = model0.predict(x0_test)[:,0] * scale
    pred_sim0.append(y_pred0_test)

    #plt.plot(x_test, y_pred0_test, label = 'HF', color = 'blue',  linewidth = 0.3)

fig = plt.figure(figsize=(6,4))

mean_sim = np.mean(np.array(pred_sim0), axis = 0).flatten()
std_sim = np.std(np.array(pred_sim0), axis = 0).flatten()
plt.plot(x_test, y_test * scale, label = 'HF', color = 'red')
plt.plot(x_test, mean_sim, label = 'mean', color = 'blue')
plt.fill_between(x_test, mean_sim - std_sim, mean_sim + std_sim, color='lightblue', alpha=0.5, label='mean $\pm$ std')
plt.plot(xhf,y * scale, 'r*')
plt.xlabel('x')
plt.legend()
plt.show()

    
#%%
#########################     MODEL 1: PREPROCESS    ##########################
###############################################################################

#Preprocess data
x1 = fidelity_order[1](xhf) / scale
x1 = x1.reshape(-1,1)

x1_test = fidelity_order[1](x_test) / scale
x1_test = x1_test.reshape(-1,1)

pred_sim1 = []
model1_list = []
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=False)
#%%
#########################     MODEL 1: TRAIN    ##########################
##########################################################################

#fig = plt.figure(figsize=(6,6))

for i in range(n_sim):
    print('sim ' + str(i+1) + ' of ' + str(n_sim))

    model1 = MultifidelityNetwork(params, input_dim = input_dim, latent_dim = latent_dim, output_dim = output_dim, prev_models = [model0_list[i]], prev_inputs = [x0])
    name = example + '/models/model1_traj_' + model_type + str(i) + '_HPO'
    if scaling:
            name = name + '_scaled'   

    #### Low-fidelity 1
    if train:
        model1.autoencoder.compile(loss='mse',optimizer=params['opt'],metrics=['mse'])
        tf.random.set_seed(seed + i)
        np.random.seed(seed + i)
        hist1 = model1.autoencoder.fit([x0, x1],y,epochs=Nepo,batch_size=Nhf,verbose=0,callbacks=[early_stopping])
    else:
        model1.load_weights(name)
    model1_list.append(model1)
    if save:
        model1.save_weights(name)

    
    #Predict
    y_pred1_test = model1.predict([x0_test, x1_test])[:,0] * scale
    pred_sim1.append(y_pred1_test)

    #plt.plot(x_test, y_pred1_test, label = 'HF', color = 'blue',  linewidth = 0.3)

fig = plt.figure(figsize=(6,4))

mean_sim = np.mean(np.array(pred_sim1), axis = 0)
std_sim = np.std(np.array(pred_sim1), axis = 0)
plt.plot(x_test, y_test * scale, label = 'HF', color = 'red')
plt.plot(x_test, mean_sim, label = 'mean', color = 'blue')
plt.fill_between(x_test, mean_sim - std_sim, mean_sim + std_sim, color='lightblue', alpha=0.5, label='mean $\pm$ std')
plt.plot(xhf,y * scale, 'r*')
plt.xlabel('x')
plt.legend()
plt.show()

#%%
#########################     MODEL 2: PREPROCESS    ##########################
###############################################################################
#Preprocess data
x2 = fidelity_order[2](xhf) / scale
x2 = x2.reshape(-1,1)

x2_test = fidelity_order[2](x_test) / scale
x2_test = x2_test.reshape(-1,1)

pred_sim2 = []
model2_list = []

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=False)

#%%
#########################     MODEL 2: TRAIN    ##########################
##########################################################################

#fig = plt.figure(figsize=(6,6))

for i in range(n_sim):
    print('sim ' + str(i+1) + ' of ' + str(n_sim))

    model2 = MultifidelityNetwork(params, input_dim = input_dim, latent_dim = latent_dim, output_dim = output_dim, prev_models = [model0_list[i], model1_list[i]], prev_inputs = [x0, x1])
    name = example + '/models/model2_traj_' + model_type + str(i) + '_HPO'
    if scaling:
            name = name + '_scaled'

    if train:
        model2.autoencoder.compile(loss='mse',optimizer=params['opt'],metrics=['mse'])
        tf.random.set_seed(seed + i)
        np.random.seed(seed + i)
        hist2 = model2.autoencoder.fit([x0, x1, x2],y,epochs=Nepo,batch_size=Nhf,verbose=0,callbacks=[early_stopping])
    else:
        model2.load_weights(name)
    model2_list.append(model2)

    if save:
        model2.save_weights(name)

    #Predict
    y_pred2_test = model2.predict([x0_test, x1_test, x2_test])[:,0] * scale
    pred_sim2.append(y_pred2_test)

    #plt.plot(x_test, y_pred2_test, label = 'HF', color = 'blue',  linewidth = 0.3)

fig = plt.figure(figsize=(6,4))

mean_sim = np.mean(np.array(pred_sim2), axis = 0)
std_sim = np.std(np.array(pred_sim2), axis = 0)
plt.plot(x_test, y_test * scale, label = 'HF', color = 'red')
plt.plot(x_test, mean_sim, label = 'mean', color = 'blue')
plt.fill_between(x_test, mean_sim - std_sim, mean_sim + std_sim, color='lightblue', alpha=0.5, label='mean $\pm$ std')
plt.plot(xhf,y * scale, 'r*')
plt.xlabel('x')
plt.legend()
plt.show()

#%%
#########################     MODEL 3    ##########################
#########################   PREPROCESS   ##########################

#Preprocess data
x3 = fidelity_order[3](xhf) / scale
x3 = x3.reshape(-1,1)

x3_test = fidelity_order[3](x_test) / scale
x3_test = x3_test.reshape(-1,1)

pred_sim3 = []
model3_list = []

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=False)

#%%
#########################     MODEL 3    ##########################
#########################      TRAIN     ##########################

#fig = plt.figure(figsize=(6,6))

for i in range(n_sim):
    print('sim ' + str(i+1) + ' of ' + str(n_sim))

    model3 = MultifidelityNetwork(params, input_dim = input_dim, latent_dim = latent_dim, output_dim = output_dim, prev_models = [model0_list[i], model1_list[i], model2_list[i]], prev_inputs = [x0, x1, x2])
    name = example + '/models/model3_traj_' + model_type + str(i) + '_HPO'
    if scaling:
            name = name + '_scaled'

    if train:
        model3.autoencoder.compile(loss='mse',optimizer=params['opt'],metrics=['mse'])
        tf.random.set_seed(seed + i)
        np.random.seed(seed + i)
        hist3 = model3.autoencoder.fit([x0, x1, x2, x3],y,epochs=Nepo,batch_size=Nhf,verbose=0,callbacks=[early_stopping])
    else:
        model3.load_weights(name)
    model3_list.append(model3)

    if save:
        model3.save_weights(name)

    #Predict
    y_pred3_test = model3.predict([x0_test, x1_test, x2_test, x3_test])[:,0] * scale
    pred_sim3.append(y_pred3_test)

    #plt.plot(x_test, y_pred3_test, label = 'HF', color = 'blue',  linewidth = 0.3)

fig = plt.figure(figsize=(6,4))

mean_sim = np.mean(np.array(pred_sim3), axis = 0)
std_sim = np.std(np.array(pred_sim3), axis = 0)
plt.plot(x_test, y_test * scale, label = 'HF', color = 'red')

plt.plot(x_test, mean_sim, label = 'mean', color = 'blue')
plt.fill_between(x_test, mean_sim - std_sim, mean_sim + std_sim, color='lightblue', alpha=0.5, label='mean $\pm$ std')
plt.plot(xhf,y * scale, 'r*')
plt.xlabel('x')
plt.legend()
plt.show()