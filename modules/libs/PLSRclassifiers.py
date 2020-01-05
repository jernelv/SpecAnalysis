from sklearn import svm
from sklearn.preprocessing import StandardScaler
import sklearn.cross_decomposition
import numpy as np
from . import PLSRCNN

def get_buttons():
    buttons=[
    {'key': 'classifier_type', 'type': 'radio:vertical:text', 'texts': ['SVC','PLS-DA','kNN','LogReg','NeuralNet'],'default': 0, 'tab': 6, 'row': 1} ,
    #SVC
    {'key': 'SVCkernel', 'type': 'radio:text', 'texts': ['Linear', 'Poly', 'Rbf', 'Sigmoid'], 'tab': 6, 'row': 1} ,
    {'key': 'SVCgamma', 'type': 'txt', 'text': 'gamma', 'default': 'auto', 'width': 5, 'tab': 6, 'row': 1} ,
    {'key': 'SVCdegree', 'type': 'txt:int:range', 'text': 'degree', 'default': '3', 'width': 2, 'tab': 6, 'row': 1} ,
    {'key': 'SVCcoef0', 'type': 'txt:float:range', 'text': 'coef0', 'default': '0.0', 'width': 4, 'tab': 6, 'row': 1} ,
    {'key': 'SVCregularisation', 'type': 'txt:float:range', 'text': 'regularisation', 'default': '1.0', 'width': 4, 'tab': 6, 'row': 1} ,
    #PLS-DA
    {'key': 'PLS-DA_latent_variables', 'type': 'txt:int:range', 'text': 'latent variables', 'default': '6,7', 'width': 5, 'tab': 6, 'row': 2} ,
    #kNN
    {'key': 'kNN_neighbours', 'type': 'txt:int:range', 'text': 'Neighbours', 'default': '3', 'width': 2, 'tab': 6, 'row': 3} ,
    #logreg
    {'key': 'LogRegpenalty', 'type': 'radio:text', 'texts': ['l2', 'l1', 'elasticnet', 'none'], 'tab': 6, 'row': 4} ,
    #NeuralNet
    #{'key': 'Clas_NN_type', 'type': 'radio:text', 'texts': ['Dense','Recurrent','Convolutional'],'default': 0, 'tab': 1, 'row': 8} ,
    {'key': 'Clas_number_of_layers', 'type': 'txt:int:range', 'text': '# of layers', 'default': '5', 'width': 5, 'tab': 6, 'row': 5} ,
    {'key': 'Clas_layer_size', 'type': 'txt:int:range', 'text': 'Size', 'default': '10', 'width': 4, 'tab': 6, 'row': 5} ,
    {'key': 'Clas_drop_frac', 'type': 'txt:float:range', 'text': 'Drop', 'default': '0.2', 'width': 5, 'tab': 6, 'row': 5} ,
    {'key': 'Clas_batch_size', 'type': 'txt:int:range', 'text': 'Batch', 'default': '1000', 'width': 7, 'tab': 6, 'row': 5} ,
    {'key': 'Clas_epochs', 'type': 'txt:int:range', 'text': 'Epochs', 'default': '2000', 'width': 6, 'tab': 6, 'row': 5} ,
    {'key': 'Clas_kernel_size', 'type': 'txt:int:range', 'text': 'C kernel size', 'default': '10', 'width': 4, 'tab': 6, 'row': 5} ,
    {'key': 'Clas_strides', 'type': 'txt:int:range', 'text': 'C strides', 'default': '1', 'width': 4, 'tab': 6, 'row': 5} ,
    {'key': 'Clas_optimizer', 'type': 'radio:text', 'texts': ['adam','sgd'],'default': 0, 'tab': 6, 'row': 5} ,
    {'key': 'Clas_learning_rate', 'type': 'txt:float:range', 'text': 'lr', 'default': '3E-5', 'width': 8, 'tab': 6, 'row': 5} ,
    {'key': 'Clas_momentum', 'type': 'txt:float:range', 'text': 'momentum', 'default': '0.9', 'width': 4, 'tab': 6, 'row': 5} ,
    ]
    return buttons

def get_classifier(common_variables,ui):
    if ui['classifier_type'] == 'SVC':
        if not ( ui['SVCgamma']=='auto'):
            ui['SVCgamma']=float(ui['SVCgamma'])
        else:
            ui['SVCgamma']=[ui['SVCgamma']]
        common_variables.keyword_lists={'kernel':[ui['SVCkernel']],'gamma':ui['SVCgamma'],'degree':ui['SVCdegree'],
            'coef0':ui['SVCcoef0'],'regularisation':ui['SVCregularisation']}
    elif ui['classifier_type'] == 'PLS-DA':
        common_variables.keyword_lists={'latent_variables':ui['PLS-DA_latent_variables']}
    elif ui['classifier_type'] == 'kNN':
        common_variables.keyword_lists={'neighbors':ui['kNN_neighbours']}
    elif ui['classifier_type'] == 'LogReg':
        common_variables.keyword_lists={'penalty':[ui['LogRegpenalty']]}
    elif ui['classifier_type'] == 'NeuralNet':
        common_variables.keyword_lists={'number_of_layers':ui['Clas_number_of_layers'],#'NN_type':[ui['Clas_NN_type']],
        'layer_size':ui['Clas_layer_size'],'drop_frac':ui['Clas_drop_frac'],'batch_size':ui['Clas_batch_size'],'epochs':ui['Clas_epochs'],
        'kernel_size':ui['Clas_kernel_size'],'strides':ui['Clas_strides'],'optimizer':[ui['Clas_optimizer']],'learning_rate':ui['Clas_learning_rate'],'momentum':ui['Clas_momentum']}

    # declare type of classifier
    common_variables.keyword_lists['classifier_type']=[ui['classifier_type']]

def get_classifier_module(reg_type,keywords,Scaling,mean_centering=True):
    if keywords['classifier_type']=='SVC':
        kernel=keywords['kernel']
        gamma=keywords['gamma']
        degree=keywords['degree']
        coef0=keywords['coef0']
        regularisation=keywords['regularisation']
        return mySupportVectorClassifier(Scaling,kernel, coef0,
            regularisation, gamma, degree,mean_centering=mean_centering)
    if keywords['classifier_type']=='PLS-DA':
        return myPLS_DA(Scaling,keywords['latent_variables'],mean_centering=mean_centering)
    if keywords['classifier_type']=='kNN':
        return mykNN(Scaling,keywords['neighbors'],mean_centering=mean_centering)
    if keywords['classifier_type']=='LogReg':
        return myLogReg(Scaling,keywords['penalty'],mean_centering=mean_centering)
    elif keywords['classifier_type']=='NeuralNet':
        number_of_layers=keywords['number_of_layers']
        layer_size=keywords['layer_size']
        drop_frac=keywords['drop_frac']
        batch_size=keywords['batch_size']
        epochs=keywords['epochs']
        kernel_size=keywords['kernel_size']
        strides=keywords['strides']
        optimizer=keywords['optimizer']
        learning_rate=keywords['learning_rate']
        momentum=keywords['momentum']
        return myCNNClassifier(Scaling, number_of_layers, layer_size, drop_frac, batch_size, epochs, kernel_size,strides=strides,mean_centering=mean_centering,
            optimizer=optimizer,learning_rate=learning_rate,momentum=momentum)

def get_correct_categorized(y1,y2):
    wrong=0
    for a,b in zip(y1,y2):
        if not a==b:
            wrong+=1
    return(1-wrong/len(y1))



class mySupportVectorClassifier:
	"""Class for support vector regression. Takes in whether or not to do scaling,
	and all parameters related to the kernel, kernel type, and regularisation.
	The fit function fits the model to the training data, while predict is used
	for predicting unknown values in the validation set."""
	type='classifier'
	def __init__(self, scale, kernel, coef0, regularisation, gamma, degree,mean_centering=True):
		#print(kernel,type(coef0),type(regularisation),type(gamma),type(degree))
		#print(gamma)
		if not 'gamma'=='auto':
			self.regr=svm.SVC(kernel=kernel.lower(),coef0=coef0,C=regularisation,gamma=gamma,degree=degree)
		else:
			self.regr=svm.SVC(kernel=kernel.lower(),coef0=coef0,C=regularisation,degree=degree)
		self.scaler=StandardScaler(copy=True, with_mean=mean_centering, with_std=scale)
	def fit(self, training, truevalues):
		self.scaler.fit(training)
		transformedTraining=self.scaler.transform(training)
		self.regr.fit(transformedTraining, truevalues)
	def predict(self, dataset):
		transformedDataset=self.scaler.transform(dataset)
		return np.rot90([self.regr.predict(transformedDataset)],3)

#pip3.6 install -U scikit-learn
#sudo -H pip3.6 install -U scikit-learn
class myPLS_DA:
    """Class for partial least-squares discriminant analysis (PLS-DA).
    Takes in whether or not to do scaling, and number of latent variables.
    The fit function fits the model to the training data, while predict is used
    for predicting unknown values in the validation set."""

    import sklearn.preprocessing
    encoder=sklearn.preprocessing.OneHotEncoder()
    type='classifier'

    def __init__(self, scale, latent_variables,mean_centering=True):
        #print(kernel,type(coef0),type(regularisation),type(gamma),type(degree))
        #print(gamma)
        self.regr=sklearn.cross_decomposition.PLSRegression(n_components=latent_variables,scale=scale)
        self.encoder=sklearn.preprocessing.OneHotEncoder()
    def fit(self, training, truevalues):
        self.encoder.fit(np.transpose(np.atleast_2d(truevalues)))
        transformedTrueValues=self.encoder.transform(np.transpose(np.atleast_2d(truevalues))).toarray()
        self.regr.fit(training, transformedTrueValues)
    def predict(self, dataset):
        pred=self.regr.predict(dataset)
        return self.encoder.inverse_transform(pred)

class mykNN:
    """Class for k-nearest neighbour classification."""

    type = 'classifier'
    def __init__(self, scale, neighbors,mean_centering=True):
        self.regr = sklearn.neighbors.KNeighborsClassifier(n_neighbors=neighbors)
        self.scaler=StandardScaler(copy=True, with_mean=mean_centering, with_std=scale)
    def fit(self, training, truevalues):
        self.scaler.fit(training)
        transformedTraining=self.scaler.transform(training)
        self.regr.fit(transformedTraining, truevalues)
    def predict(self, dataset):
        transformedDataset=self.scaler.transform(dataset)
        return np.rot90([self.regr.predict(transformedDataset)],3)

class myLogReg:
    """Class for logistic regression."""

    type = 'classifier'
    def __init__(self, scale, penalty,mean_centering=True):
        self.regr = sklearn.linear_model.LogisticRegression(penalty=penalty.lower())
        self.scaler = StandardScaler(copy=True, with_mean=mean_centering, with_std=scale)
    def fit(self, training, truevalues):
        self.scaler.fit(training)
        transformedTraining=self.scaler.transform(training)
        self.regr.fit(transformedTraining, truevalues)
    def predict(self, dataset):
        transformedDataset=self.scaler.transform(dataset)
        return np.rot90([self.regr.predict(transformedDataset)],3)

class myCNNClassifier:
    type='classifier'
    def __init__(self, scaling, number_of_layers, layer_size, drop_frac, batch_size, epochs,kernel_size,strides=1,mean_centering=True,optimizer='adam',learning_rate=3*10**-5,momentum=0.9):
        self.encoder=sklearn.preprocessing.OneHotEncoder()
        self.scaler=StandardScaler(copy=True, with_mean=mean_centering, with_std=scaling)
        self.number_of_layers=number_of_layers
        self.layer_size=layer_size
        self.drop_frac=drop_frac
        self.batch_size=batch_size
        self.epochs=epochs
        self.kernel_size=kernel_size
        self.strides=strides
        self.optimizer=optimizer
        self.learning_rate=learning_rate
        self.momentum=momentum
    def fit(self, training, truevalues):
        self.encoder.fit(np.transpose(np.atleast_2d(truevalues)))
        transformedTrueValues=self.encoder.transform(np.transpose(np.atleast_2d(truevalues))).toarray()
        self.scaler.fit(training)
        transformedTraining = self.scaler.transform(training)
        self.neural_net=PLSRCNN.myNeuralNet(self.number_of_layers, self.layer_size,
            self.drop_frac, self.batch_size, self.epochs,self.kernel_size,strides=self.strides,num_out=transformedTrueValues.shape[1],
            optimizer=self.optimizer,learning_rate=self.learning_rate,momentum=self.momentum)
        self.neural_net.fit(transformedTraining, np.rot90((transformedTrueValues)))
    def predict(self, dataset):
        transformedDataset=self.scaler.transform(dataset)
        return self.encoder.inverse_transform(self.neural_net.predict(transformedDataset))
