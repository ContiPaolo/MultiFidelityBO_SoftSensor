#%%
import tensorflow.keras.backend as K

from tensorflow.keras.regularizers import l2
from hyperopt import STATUS_OK, tpe, Trials, hp, fmin

import numpy as np

from tensorflow.keras.optimizers import Adam,Nadam,Adamax

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Add, Lambda, BatchNormalization
from tensorflow.keras.layers import concatenate
import tensorflow as tf

#%%

class MultifidelityNetwork(tf.keras.Model):
    def __init__(self, params, input_dim, latent_dim, output_dim, prev_models = [], prev_inputs = []):
        super().__init__()
        self.build_model(params, input_dim, latent_dim, output_dim, prev_models, prev_inputs)


    def build_model(self, params, input_dim, latent_dim, output_dim, prev_models, prev_inputs):
        '''
        Construct the model by calling the encoder, decoder and output summation functions.
        :param params: dictionary with neural network parameters
        :param input_dim: dimension of the input data
        :param latent_dim: dimension of the latent space
        :param output_dim: dimension of the output space
        :param prev_models: list of previous models
        :param prev_inputs: list of previous inputs
        '''
        #Process input data
        data_input, prev_data_inputs = self.process_input_data(input_dim, prev_inputs) 

        #Encode input data
        latent = self.build_encoder(params, latent_dim, data_input)

        #Concatenate latent variables
        latent_tot = self.concatenate_latents(latent, prev_data_inputs, prev_models, params['concatenate'])

        #Decode latent variables
        decoder_output = self.build_decoder(params, latent_tot, output_dim)

        #Output summation
        #outputs = self.output_summation(decoder_output, latent_tot_prev, prev_models, prev_inputs)
        outputs = self.sum_output(decoder_output, prev_data_inputs, prev_models)

        self.encoder = tf.keras.Model(inputs=data_input, outputs=latent)
        self.decoder = tf.keras.Model(inputs=latent_tot, outputs=decoder_output)
        self.autoencoder = tf.keras.Model(inputs=prev_data_inputs + [data_input], outputs=outputs)


    def process_input_data(self, input_dim, prev_inputs):
        '''
        Create input tensors to create the model

        :param input_dim: dimension of the input data
        :param prev_inputs: list of previous inputs
        :return: current input data tensor and list of previous input data tensors
        '''
        data_input = Input(shape=(input_dim,))
        prev_data_inputs = []
        for prev_input in prev_inputs:
            prev_data_input = Input(shape=(prev_input.shape[1],))
            prev_data_input.trainable = False
            prev_data_inputs.append(prev_data_input)
        return data_input, prev_data_inputs


    def build_encoder(self, params, latent_dim, encoder_input):
        '''
        Define the encoder model which processes and maps the input data to the latent space
        :param params: dictionary with neural network parameters
        :param latent_dim: dimension of the latent space
        :param encoder_input: current input data
        :return: current latent variable (output of the encoder)
        '''
        if len(params['layers_encoder']) == 0:
            latent = encoder_input
        else:
            h = encoder_input
            for nodes in params['layers_encoder']:
                h = Dense(nodes,  activation = params['activation'], kernel_regularizer=l2(params['l2weight']), kernel_initializer=params['kernel_init'])(h)
            latent = Dense(latent_dim, activation = 'linear', kernel_regularizer=l2(params['l2weight']), kernel_initializer=params['kernel_init'])(h)
        return latent


    def concatenate_latents(self, latent, prev_data_inputs, prev_models, concatenate_inputs = False):
        '''
        Concatenate the current latent variable with the previous ones.
        :param latent: current latent variable
        :param prev_data_inputs: list of previous input data
        :param prev_models: list of previous models
        :return: concatenated latent variables (previous + current)
        '''  
        if not concatenate_inputs:
            print('Not concatenating')
            return latent
      
        if len(prev_models) == 0:
            #if this is the first level just return the current latent variable
            latent_tot = latent
        else:
            latent_prevs = [latent]
            for prev_input, prev_model in zip(prev_data_inputs, prev_models):
                #make previous models non-trainable
                prev_model.encoder.trainable = False
                #compute latent of previous models
                latent_prevs.append(prev_model.encoder(prev_input))
            #concatenate all latents
            latent_tot = concatenate(latent_prevs)
        return latent_tot
        


    def build_decoder(self, params, latent_tot, output_dim):
        '''
        Define the decoder model which takes the concatenated latent variables and maps them to the output space
        :param latent_tot: concatenated latent variables
        :param output_dim: dimension of the output space
        '''
        h = latent_tot
        for nodes in params['layers_decoder']:
            h = Dense(nodes,  activation = params['activation'], kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(h)
        decoder_output = Dense(output_dim, activation = 'linear', kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(h)
        return decoder_output

    
    def sum_output(self, decoder_output, prev_data_inputs, prev_models):
        '''
        Sum the output of the current model with the output of the previous ones.

        :param decoder_output: output of the current model
        :param prev_models: list of previous models
        :param prev_inputs: list of previous inputs
        '''
        if len(prev_models) == 0:
            #if this is the first level just return the current output
            outputs = decoder_output
        else:
            #retrieve last model and its input
            prev_model, prev_input = prev_models[-1], prev_data_inputs[:len(prev_models)]
            #compute output of previous model
            output_prev = prev_model(prev_input)
            #make previous model non-trainable
            output_prev.trainable = False
            #sum current decoder output with previous output
            outputs = Add()([output_prev, decoder_output])
        return outputs

    def call(self, x):
        return self.autoencoder(x)

    def predict(self, x_test):
        return self(x_test)

    def save_weights(self, save_path):
        """
        Save the weights of the model to a file.

        :param save_path: Path to save the weights to.
        """
        print('Saving weights to: ' + save_path)
        self.autoencoder.save_weights(save_path)

    def load_weights(self, load_path):
        """
        Load the weights from a saved file and assign them to the model.

        :param load_path: Path to the saved weights file.
        """
        print('Loading weights from: ' + load_path)
        self.autoencoder.load_weights(load_path)

#%%
def sliding_windows(data_input, data_output, seq_length, freq=1):
    x = []
    y = []

    for i in range(data_input.shape[0]):
        for j in range(0, data_input.shape[1] - seq_length, freq):
            _x = data_input[i, j:(j + seq_length), :]
            _y = data_output[i, j:(j + seq_length), :]
            x.append(_x)
            y.append(_y)

    return np.array(x), np.array(y)