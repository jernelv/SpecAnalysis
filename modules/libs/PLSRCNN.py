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
    def __init__(self, number_of_layers, layer_size, drop_frac, batch_size, epochs,kernel_size,strides=1,verbose=True, num_out=1,optimizer='adam',learning_rate=3*10**-5,momentum=0.9):
        self.number_of_layers=number_of_layers
        self.layer_size=layer_size
        self.drop_frac=drop_frac
        self.batch_size=batch_size
        self.epochs=epochs
        self.verbose=verbose
        self.kernel_size=kernel_size
        self.strides=strides
        self.seed=42
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.optimizer=optimizer
        self.num_out=num_out
        # clear_session should clear the memory used, but keras 2.2.4 has a memory leak when used in this way
        #this can be solved by downgrading to keras 2.1.6
        import keras
        from keras import backend
        keras.backend.clear_session()
        self.session=keras.backend.get_session()
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
                np.atleast_2d(scaled_y), epochs=self.epochs, batch_size=self.batch_size,verbose=self.verbose)
        for layer in self.model.layers:
            layer.trainable = False
        self.last_layer.trainable = True
        self.model.compile(self.optimizer_object, loss='mse')
        import keras
        # set up for getting values at nodes
        #outputs = [layer.output for layer in self.model.layers]
        #self.functors = [keras.backend.function([self.model.input, keras.backend.learning_phase()], [out]) for out in outputs]    # evaluation functions
                                 # input placeholder
    def make_model(self,input_size):
        import numpy.random
        numpy.random.seed(self.seed)
        import tensorflow
        tensorflow.set_random_seed(self.seed)
        import keras
        layer = keras.layers.Conv1D
        #wrapper = keras.layers.Bidirectional
        self.model = keras.models.Sequential()
        #self.model.add(keras.layers.Dense(self.layer_size, activation='linear',input_dim=input_size))
        #self.model.add(wrapper(layer(self.layer_size, return_sequences=(self.number_of_layers > 1)),input_shape=(input_size, 1)))
        self.model.add(layer(filters=self.layer_size,kernel_size=self.kernel_size, activation='relu',input_shape=(input_size, 1),strides=self.strides))
        #self.model.add(keras.layers.ReLU())
        self.add_dropout(self.drop_frac)
        for i in range(1, self.number_of_layers):
            #self.model.add(wrapper(layer(self.layer_size, return_sequences=(i < self.number_of_layers - 1)),input_shape=(input_size, 1)))
            self.model.add(layer(filters=self.layer_size,kernel_size=self.kernel_size, activation='relu',strides=self.strides))
            #self.model.add(keras.layers.Dense(self.layer_size, activation='linear'))
            #self.model.add(keras.layers.ReLU())
            self.add_dropout( self.drop_frac)
        #add output layers
        '''
        self.model.add(keras.layers.MaxPooling1D(pool_size=2))
        self.model.add(keras.layers.Dense(100,activation='relu'))
        '''
        self.model.add(keras.layers.Flatten())
        self.last_layer=keras.layers.Dense(self.num_out)
        self.model.add(self.last_layer)
        if self.optimizer=='adam':
            self.optimizer_object=keras.optimizers.Adam(self.learning_rate)
        else:
            self.optimizer_object=keras.optimizers.SGD(lr=self.learning_rate, momentum=self.momentum)
        self.model.compile(self.optimizer_object, loss='mse')
        # set up to get weights
        inp=self.model.input
        outputs = [layer.output for layer in self.model.layers]          # all layer outputs
        self.functor = keras.backend.function([inp, keras.backend.learning_phase()], outputs )   # evaluation function
        # Testing
        print('Created network')
        #self.activation=my_relu
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

    def scramble_last_layer(self):
        self.last_layer.kernel.initializer.run(session=self.session)
        return
    def retrain_only_last(self,training):
        '''for layer in self.model.layers:
            layer.trainable = False
        self.last_layer.trainable = True
        self.model.compile(self.optimizer_object, loss='mse')'''
        y_rot=np.rot90(np.atleast_2d(training.Y),-1)
        scaled_y=self.y_scaler.fit_transform(y_rot)
        #fit model
        self.model.fit(np.atleast_3d(training.X),np.atleast_2d(scaled_y), epochs=1, batch_size=self.batch_size,verbose=False)
        return
    def get_last_layer_weights(self):
        return self.last_layer.get_weights()
    def stability_selection_pass(self,training):
        self.scramble_last_layer()
        scrambled_weights = np.array(self.get_last_layer_weights())[0][:,0]
        self.retrain_only_last(training)
        retrained_weights = np.array(self.get_last_layer_weights())[0][:,0]
        #positively_changed_weights=np.zeros(retrained_weights.shape)
        positively_changed_weights=retrained_weights-scrambled_weights#>0]=1
        return positively_changed_weights
    def do_stability_selection(self,run):
        positively_changed_weights=self.stability_selection_pass(run.last_Xval_case.T)
        for i in range(100-1):
            positively_changed_weights+=self.stability_selection_pass(run.last_Xval_case.T)
            print(str(i+1)+' of 100')
            run.draw()
        ui=run.ui
        ui['fig_per_row']=int(run.frame.buttons['fig_per_row'].get())
        ui['max_plots']=int(run.frame.buttons['max_plots'].get())
        import fns
        ax=fns.add_axis(run.common_variables.fig,ui['fig_per_row'],ui['max_plots'])
        positively_changed_weights=positively_changed_weights.reshape((-1,self.layer_size))

        wavenum=run.last_complete_case.wavenumbers
        dl=(len(wavenum)-len(positively_changed_weights))//2
        wavenum=wavenum[dl:-dl]
        for i,a in enumerate(positively_changed_weights[0]):
            ax.plot(wavenum,positively_changed_weights[:,i])#,label=i)
        ax.invert_xaxis()

        #ax.legend()
        #ax.plot(positively_changed_weights)
        run.draw()

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
