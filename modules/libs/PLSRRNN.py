import os
'''import tensorflow as tf
#this is useful if the user somhow runs tf on a GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #set to allow dynamic allocation of memory on gpu
session = tf.Session(config=config)'''
import numpy as np
from sklearn.preprocessing import StandardScaler

class myNeuralNet:
    """Class for regression with deep neural net."""
    def __init__(self, number_of_layers, layer_size, drop_frac, batch_size, epochs):
        self.number_of_layers=number_of_layers
        self.layer_size=layer_size
        self.drop_frac=drop_frac
        self.batch_size=batch_size
        self.epochs=epochs
        self.seed=42
        self.learning_rate=3*10**-5
        # clear_session should clear the memory used, but keras 2.2.4 has a memory leak when used in this way
        #this can be solved by downgrading to keras 2.1.6
        import keras
        from keras import backend
        keras.backend.clear_session()
        #tf.reset_default_graph()
        '''old_session=keras.backend.get_session()
        try:
            old_session.close()
        except:
            None
        self.session = tf.Session()
        keras.backend.set_session(session)'''

    def fit(self, data, data_Y):
        #make model
        input_size=data.shape[-1]
        self.make_model(input_size)
        #rotate and scale y data
        self.y_scaler = StandardScaler()
        y_rot=np.rot90(np.atleast_2d(data_Y),-1)
        scaled_y=self.y_scaler.fit_transform(y_rot)
        #fit model
        self.history = self.model.fit(np.atleast_3d(data),
                np.atleast_2d(scaled_y), epochs=self.epochs, batch_size=self.batch_size,verbose=1)
        import keras
        # set up for getting values at nodes
        #outputs = [layer.output for layer in self.model.layers]
        #self.functors = [keras.backend.function([self.model.input, keras.backend.learning_phase()], [out]) for out in outputs]    # evaluation functions
                                 # input placeholder
    def make_model(self,input_size):
        import keras
        layer = keras.layers.LSTM
        wrapper = keras.layers.Bidirectional
        self.model = keras.models.Sequential()
        #self.model.add(keras.layers.Dense(self.layer_size, activation='linear',input_dim=input_size))
        self.model.add(wrapper(layer(self.layer_size, return_sequences=(self.number_of_layers > 1)),
                  input_shape=(input_size, 1)))
        #self.model.add(keras.layers.ReLU())
        self.add_dropout(self.drop_frac)
        for i in range(1, self.number_of_layers):
            self.model.add(wrapper(layer(self.layer_size, return_sequences=(i < self.number_of_layers - 1)),
                      input_shape=(input_size, 1)))
            #self.model.add(keras.layers.Dense(self.layer_size, activation='linear'))
            #self.model.add(keras.layers.ReLU())
            self.add_dropout( self.drop_frac)
        #add output layers
        self.model.add(keras.layers.Dense(1))
        self.model.compile(keras.optimizers.Adam(self.learning_rate), loss='mse')
        # set up to get weights
        inp=self.model.input
        outputs = [layer.output for layer in self.model.layers]          # all layer outputs
        self.functor = keras.backend.function([inp, keras.backend.learning_phase()], outputs )   # evaluation function
        # Testing
        print('Created network')
        self.activation=my_relu
    def predict(self,data):
        predicted=self.model.predict(np.atleast_3d(data))
        unscaled_predicted=self.y_scaler.inverse_transform(predicted)
        return unscaled_predicted

    def get_weights(self):
        weights=[]
        for layer in self.model.layers:
            weights.append(layer.get_weights()) # list of numpy arrays
        return weights

    def get_values(self,data):
        import keras
        #layer_outs = [func([data, 0]) for func in self.functors]
        self.layer_outs = self.functor([data, 0.])
        return(self.layer_outs)
    def add_dropout(self, value):
        import keras
        if value > 0:
            return self.model.add(keras.layers.Dropout(value))
        else:
            pass
def my_relu(inp,pivot=''):
    if pivot=='':
        pivot=inp
    if type(inp)==np.ndarray:
        #print(np.sum(pivot<0))
        inp[pivot<0]=0
    else:
        if pivot<0:
            inp=0
    return inp
