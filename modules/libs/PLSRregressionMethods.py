
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.datasets import make_regression
import numpy as npNone
from sklearn import svm
from sklearn.linear_model import ElasticNet
from . import PLSRNN
from . import PLSRRNN
from . import PLSRCNN
from modules import PLSR
from . import PLSRregressionVisualization
import copy
import numpy as np
from . import PLSRclassifiers

def get_buttons():
    buttons=[
    {'key': 'PLSRtab1name', 'type': 'tabname', 'text': 'Regression Methods', 'tab': 1} ,
    #{'key': 'RegressionL1', 'type': 'label', 'text': 'Regression options: ', 'tab': 1, 'row': 1} ,
    {'key': 'reg_type', 'type': 'radio:vertical:text', 'texts': ['None','MLR','PLSR', 'PCR', 'Tree', 'SVR', 'ElasticNet','NeuralNet','Classifier'],'default': 2, 'tab': 1, 'row': 1} ,
    #MLR:
    #PLSR:
    {'key': 'Latent variables', 'type': 'txt:int:range', 'text': 'Latent variables', 'default': '2,3', 'width': 4, 'tab': 1, 'row': 3} ,
    {'key': 'plot_components_PLSR', 'type': 'click', 'text': 'Plot latent variables', 'bind': PLSRregressionVisualization.plot_components_PLSR, 'tab': 1, 'row': 3} ,
    #PCR
    {'key': 'Components', 'type': 'txt:int:range', 'text': 'components', 'default': '6,7', 'width': 4, 'tab': 1, 'row': 4} ,
    {'key': 'plot_components_PCR', 'type': 'click', 'text': 'Plot components', 'bind': PLSRregressionVisualization.plot_components_PCR, 'tab': 1, 'row': 4} ,
    {'key': 'plot_PCR_scatter', 'type': 'click', 'text': 'Plot scatter plot of weights', 'bind': PLSRregressionVisualization.plot_PCR_scatter, 'tab': 1, 'row': 4} ,
    {'key': 'PCR_scatter_X', 'type': 'txt:int', 'text': 'X', 'default': '1', 'width': 2, 'tab': 1, 'row': 4} ,
    {'key': 'PCR_scatter_Y', 'type': 'txt:int', 'text': 'Y', 'default': '2', 'width': 2, 'tab': 1, 'row': 4} ,
    #tree
    {'key': 'Depth', 'type': 'txt:int:range', 'text': 'Tree depth start', 'default': '10', 'width': 4, 'tab': 1, 'row': 5} ,
    {'key': 'n_estimators', 'type': 'txt:int:range', 'text': 'n_estimators', 'default': '200', 'width': 4, 'tab': 1, 'row': 5} ,
    {'key': 'plot_feature_importance', 'type': 'click', 'text': 'Plot feature importance', 'bind':PLSRregressionVisualization.plot_feature_importance, 'tab': 1, 'row': 5} ,
    #support vector regression
    {'key': 'kernel', 'type': 'radio:text', 'texts': ['Linear', 'Poly', 'Rbf', 'Sigmoid'], 'tab': 1, 'row': 6} ,
    {'key': 'gamma', 'type': 'txt', 'text': 'gamma', 'default': 'auto', 'width': 5, 'tab': 1, 'row': 6} ,
    {'key': 'degree', 'type': 'txt:int:range', 'text': 'degree', 'default': '3', 'width': 2, 'tab': 1, 'row': 6} ,
    {'key': 'coef0', 'type': 'txt:float:range', 'text': 'coef0', 'default': '0.0', 'width': 4, 'tab': 1, 'row': 6} ,
    {'key': 'regularisation', 'type': 'txt:float:range', 'text': 'regularisation', 'default': '1.0', 'width': 4, 'tab': 1, 'row': 6} ,
    #elastic net
    {'key': 'l1_ratio', 'type': 'txt:float:range', 'text': 'l1_ratio', 'default': '0.5', 'width': 4, 'tab': 1, 'row': 7} ,
    #neural net
    {'key': 'NN_type', 'type': 'radio:text', 'texts': ['Dense','Recurrent','Convolutional'],'default': 0, 'tab': 1, 'row': 8} ,
    {'key': 'number_of_layers', 'type': 'txt:int:range', 'text': '# of layers', 'default': '5', 'width': 3, 'tab': 1, 'row': 8} ,
    {'key': 'layer_size', 'type': 'txt:int:range', 'text': 'Size', 'default': '10', 'width': 4, 'tab': 1, 'row': 8} ,
    {'key': 'drop_frac', 'type': 'txt:float:range', 'text': 'Drop', 'default': '0.2', 'width': 5, 'tab': 1, 'row': 8} ,
    {'key': 'batch_size', 'type': 'txt:int:range', 'text': 'Batch', 'default': '1000', 'width': 5, 'tab': 1, 'row': 8} ,
    {'key': 'epochs', 'type': 'txt:int:range', 'text': 'Epochs', 'default': '2000', 'width': 6, 'tab': 1, 'row': 8} ,
    {'key': 'kernel_size', 'type': 'txt:int:range', 'text': 'C kernel size', 'default': '10', 'width': 4, 'tab': 1, 'row': 8} ,
    {'key': 'strides', 'type': 'txt:int:range', 'text': 'C strides', 'default': '1', 'width': 4, 'tab': 1, 'row': 8} ,
    {'key': 'plot_node_correlations', 'type': 'click', 'text': 'Plot nodes', 'bind': PLSRregressionVisualization.plot_node_correlations, 'tab': 1, 'row': 8} ,
    {'key': 'optimizer', 'type': 'radio:text', 'texts': ['adam','sgd'],'default': 0, 'tab': 1, 'row': 8} ,
    {'key': 'learning_rate', 'type': 'txt:float:range', 'text': 'lr', 'default': '3E-5', 'width': 8, 'tab': 1, 'row': 8} ,
    {'key': 'momentum', 'type': 'txt:float:range', 'text': 'momentum', 'default': '0.9', 'width': 4, 'tab': 1, 'row': 8} ,
    #RNN

    {'key': 'PLSRtab6name', 'type': 'tabname', 'text': 'Classifier methods Methods', 'tab': 6} ,
    {'key': 'ClassifiersL1', 'type': 'label', 'text': 'To use a classifier, set "Classifier" as regression option in tab 2', 'tab': 6, 'row': 0} ,


    ]
    return buttons

def get_relevant_keywords(common_variables,ui):
    '''
    this section makes the relevant keyword lists.
    Each keyword should represent a list of possible values
    All permutations of keywords are then generated by generate_keyword_cases()
    keywords that are strings therefore need to be elements in a list,
        otherwise each char will be interpreted as a different cases
    '''
    if ui['reg_type'] == 'MLR':
        common_variables.keyword_lists={}
    elif ui['reg_type'] == 'PLSR':
        common_variables.keyword_lists={'Latent variables':ui['Latent variables']}
    elif ui['reg_type'] == 'PCR':
        common_variables.keyword_lists={'Components':ui['Components']}
    elif ui['reg_type'] == 'Tree':
        common_variables.keyword_lists={'n_estimators':ui['n_estimators'],'Depth':ui['Depth']}
    elif ui['reg_type'] == 'SVR':#ui['classifier_type'] == 'SVC'):
        if not ( ui['gamma']=='auto'):
            ui['gamma']=float(ui['gamma'])
        else:
            ui['gamma']=[ui['gamma']]
        common_variables.keyword_lists={'kernel':[ui['kernel']],'gamma':ui['gamma'],'degree':ui['degree'],
            'coef0':ui['coef0'],'regularisation':ui['regularisation']}
    elif ui['reg_type'] == 'ElasticNet':
        common_variables.keyword_lists={'l1_ratio':ui['l1_ratio']}
    elif ui['reg_type'] == 'NeuralNet':
        common_variables.keyword_lists={'number_of_layers':ui['number_of_layers'],'NN_type':[ui['NN_type']],
        'layer_size':ui['layer_size'],'drop_frac':ui['drop_frac'],'batch_size':ui['batch_size'],'epochs':ui['epochs'],'kernel_size':ui['kernel_size'],
        'strides':ui['strides'],'optimizer':[ui['optimizer']],'learning_rate':ui['learning_rate'],'momentum':ui['momentum']}
    if ui['reg_type'] == 'Classifier':
        PLSRclassifiers.get_classifier(common_variables,ui)

	#{'key': 'mean_centering', 'type': 'radio:text', 'texts': ['No mean centering', 'Mean centering','Try all'], 'tab': 0, 'row': 2} ,
    if ui['mean_centering']=='No mean centering':
        common_variables.keyword_lists['mean_centering']=[False]
    elif ui['mean_centering']=='Mean centering':
        common_variables.keyword_lists['mean_centering']=[True]
    else:
        common_variables.keyword_lists['mean_centering']=[False,True]

    #{'key': 'scaling', 'type': 'radio:text', 'texts': ['No scaling', 'Scaling','Try all'], 'tab': 0, 'row': 2} ,
    if ui['scaling']=='No scaling':
        common_variables.keyword_lists['scaling']=[False]
    elif ui['scaling']=='Scaling':
        common_variables.keyword_lists['scaling']=[True]
    else:
        common_variables.keyword_lists['scaling']=[False,True]
    #common_variables.keyword_lists['scaling']=[ui['scaling']]
    #common_variables.keyword_lists['mean_centering']=[ui['mean_centering']]

    '''elif ui['reg_type'] == 'Neural Net with bidirectional long short term memory':
        common_variables.keyword_lists={'number_of_layers':ui['number_of_layers'],
        'layer_size':ui['layer_size'],'drop_frac':ui['drop_frac'],'batch_size':ui['batch_size'],'epochs':ui['epochs']}
    elif ui['reg_type'] == 'Convolutional neural net':
        common_variables.keyword_lists={'number_of_layers':ui['number_of_layers'],
        'layer_size':ui['layer_size'],'drop_frac':ui['drop_frac'],'batch_size':ui['batch_size'],'epochs':ui['epochs']}'''

def generate_keyword_cases(keyword_lists):
    keys = list(keyword_lists.keys())
    for key in keys:
        keyword_lists[key]=list(keyword_lists[key])
    new_combinations=[{}]
    for key in keys:
        cur_keyword_list=keyword_lists[key]
        old_combinations=new_combinations
        new_combinations=[]
        for i, di in enumerate(old_combinations):
            for j in range(len((cur_keyword_list))-1):
                new_combinations.append(copy.deepcopy(di))
                new_combinations[-1][key]=cur_keyword_list[j]
            new_combinations.append(di)
            new_combinations[-1][key]=cur_keyword_list[-1]
    return new_combinations

def getRegModule(reg_type,keywords):
	"""Function used to get the regression type and any associated parameters."""
	Scaling=keywords['scaling']
	mean_centering=keywords['mean_centering']
	if reg_type=='MLR':
		return mlr(Scaling)
	elif reg_type=='PLSR':
		latent_variables=keywords['Latent variables']
		reg_module=PLSRegression(n_components=latent_variables,scale=Scaling)
		reg_module.type='regression'
		return reg_module
	elif reg_type=='PCR':
		components=keywords['Components']
		return pcr(components,Scaling,mean_centering=mean_centering)
	elif reg_type=='Tree':
		depth=keywords['Depth']
		n_estimators=keywords['n_estimators']
		return myRandomForestRegressor(depth,Scaling,n_estimators=n_estimators,mean_centering=mean_centering)
	elif reg_type=='SVR':
		kernel=keywords['kernel']
		gamma=keywords['gamma']
		degree=keywords['degree']
		coef0=keywords['coef0']
		regularisation=keywords['regularisation']
		return mySupportVectorRegressor(Scaling,kernel, coef0,
					regularisation, gamma, degree,mean_centering=mean_centering)
	elif reg_type=='ElasticNet':
		l1_ratio=keywords['l1_ratio']
		return myElasticNetRegressor(Scaling, l1_ratio,mean_centering=mean_centering)
	elif reg_type=='NeuralNet':
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
		if keywords['NN_type']=='Dense':
			return myNeuralNetRegressor(Scaling, number_of_layers, layer_size, drop_frac, batch_size, epochs,mean_centering=mean_centering)
		elif keywords['NN_type']=='Recurrent':
			return myRNNRegressor(Scaling, number_of_layers, layer_size, drop_frac, batch_size, epochs,mean_centering=mean_centering)
		else: #Convolutional
			return myCNNRegressor(Scaling, number_of_layers, layer_size, drop_frac, batch_size, epochs, kernel_size,strides=strides,mean_centering=mean_centering,
                optimizer=optimizer,learning_rate=learning_rate,momentum=momentum)
	elif reg_type=='Classifier':
		return PLSRclassifiers.get_classifier_module(reg_type,keywords)

class mlr:
	"""Implementation of multiple linear regression based on linear regression,
	which takes in whether or not to scale the data. The fit and predict functions
	are used to fit the model to training data and to predict the unknown
	validation data, respectively."""
	type='regression'
	def __init__(self, scale,mean_centering=True):
		self.linreg=LinearRegression()
		self.scaler=StandardScaler(copy=True, with_mean=mean_centering, with_std=scale)
	def fit(self, training,truevalues):
		self.scaler.fit(training)
		transformedTraining=self.scaler.transform(training) #scale(training)
		self.linreg.fit(transformedTraining, truevalues)

	def predict(self, dataset):
		transformedDataset=self.scaler.transform(dataset)
		return np.rot90([self.linreg.predict(transformedDataset)],3)
class pcr:
	"""Implementation of principal component regression as a combination of the
	PCA and linear regression methods from scipy. Takes in the number of principal
	components and whether or not to scale as parameters. The fit and predict
	functions are used to fit the model to training data and to predict the
	unknown validation data."""
	type='regression'
	def __init__(self, components, scale,mean_centering=True):
		self.linreg=LinearRegression()
		self.pca=PCA(n_components=components)
		self.components=components
		self.scaler=StandardScaler(copy=True, with_mean=mean_centering, with_std=scale)

	def fit(self, training,truevalues):
		self.scaler.fit(training)
		transformedTraining=self.scaler.transform(training) #scale(training)
		X_reduced = self.pca.fit_transform((transformedTraining))
		self.linreg.fit(X_reduced[:,:self.components], truevalues)

	def get_X_reduced(self, dataset):
		transformedDataset=self.scaler.transform(dataset)
		X_reduced = self.pca.transform(transformedDataset) #scale(dataset)
		return X_reduced

	def predict(self, dataset):
		X_reduced=self.get_X_reduced(dataset)
		return np.rot90([self.linreg.predict(X_reduced[:,:self.components])],3)
class myRandomForestRegressor:
	"""Class for random forest regression. Takes in tree branching depth,
	the number of trees to make, and whether or not to scale the data. The fit
	and predict functions are used to fit the model to training data and to predict
	unknown validation data, respectively."""
	type='regression'
	def __init__(self, depth, scale, n_estimators=200,mean_centering=True):
		self.regr=RandomForestRegressor(max_depth=depth, #random_state=0,
			n_estimators=n_estimators)
		self.scaler=StandardScaler(copy=True, with_mean=mean_centering, with_std=scale)
	def fit(self, training,truevalues):
		self.scaler.fit(training)
		transformedTraining=self.scaler.transform(training) #scale(training)
		self.regr.fit(transformedTraining, truevalues)
		'''RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
		   max_features='auto', max_leaf_nodes=None,
		   min_impurity_decrease=0.0, min_impurity_split=None,
		   min_samples_leaf=1, min_samples_split=2,
		   min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
		   oob_score=False, random_state=0, verbose=0, warm_start=False)'''
	def predict(self, dataset):
		transformedDataset=self.scaler.transform(dataset)
		#print(self.regr.predict(transformedDataset))
		return np.rot90([self.regr.predict(transformedDataset)],3)

class mySupportVectorRegressor:
	"""Class for support vector regression. Takes in whether or not to do scaling,
	and all parameters related to the kernel, kernel type, and regularisation.
	The fit function fits the model to the training data, while predict is used
	for predicting unknown values in the validation set."""
	type='regression'
	def __init__(self, scale, kernel, coef0, regularisation, gamma, degree,mean_centering=True):
		#print(kernel,type(coef0),type(regularisation),type(gamma),type(degree))
		#print(gamma)
		if not 'gamma'=='auto':
			self.regr=svm.SVR(kernel=kernel.lower(),coef0=coef0,C=regularisation,gamma=gamma,degree=degree)
		else:
			self.regr=svm.SVR(kernel=kernel.lower(),coef0=coef0,C=regularisation,degree=degree)
		self.scaler=StandardScaler(copy=True, with_mean=mean_centering, with_std=scale)
	def fit(self, training, truevalues):
		self.scaler.fit(training)
		transformedTraining=self.scaler.transform(training)
		self.regr.fit(transformedTraining, truevalues)
	def predict(self, dataset):
		transformedDataset=self.scaler.transform(dataset)
		return np.rot90([self.regr.predict(transformedDataset)],3)

class myElasticNetRegressor:
	"""Class for elastic net regression. Takes in whether or not to do scaling,
	and the l1_ratio parameter. The fit and predict functions are used for fitting
	the model to known data and to predict on unknown data, respectively."""
	type='regression'
	def __init__(self, scale, l1_ratio, mean_centering=True):
		self.regr=ElasticNet(l1_ratio=l1_ratio, random_state=1)
		self.scaler=StandardScaler(copy=True, with_mean=mean_centering, with_std=scale)
	def fit(self, training, truevalues):
		self.scaler.fit(training)
		transformedTraining = self.scaler.transform(training)
		self.regr.fit(transformedTraining, truevalues)
	def predict(self, dataset):
		transformedDataset=self.scaler.transform(dataset)
		return np.rot90([self.regr.predict(transformedDataset)], 3)

class myNeuralNetRegressor:
	type='regression'
	def __init__(self, scaling, number_of_layers, layer_size, drop_frac, batch_size, epochs,mean_centering=True):
		self.neural_net=PLSRNN.myNeuralNet(number_of_layers, layer_size, drop_frac, batch_size, epochs)
		self.scaler=StandardScaler(copy=True, with_mean=mean_centering, with_std=scaling)
	def fit(self, training, truevalues):
		self.scaler.fit(training)
		transformedTraining = self.scaler.transform(training)
		self.neural_net.fit(transformedTraining, truevalues)
	def predict(self, dataset):
		transformedDataset=self.scaler.transform(dataset)
		return np.rot90([self.neural_net.predict(transformedDataset).reshape(-1)], 3)

class myRNNRegressor:
	type='regression'
	def __init__(self, scaling, number_of_layers, layer_size, drop_frac, batch_size, epochs,mean_centering=True):
		self.neural_net=PLSRRNN.myNeuralNet(number_of_layers, layer_size, drop_frac, batch_size, epochs)
		self.scaler=StandardScaler(copy=True, with_mean=mean_centering, with_std=scaling)
	def fit(self, training, truevalues):
		self.scaler.fit(training)
		transformedTraining = self.scaler.transform(training)
		self.neural_net.fit(transformedTraining, truevalues)
	def predict(self, dataset):
		transformedDataset=self.scaler.transform(dataset)
		return np.rot90([self.neural_net.predict(transformedDataset).reshape(-1)], 3)

class myCNNRegressor:
	type='regression'
	def __init__(self, scaling, number_of_layers, layer_size, drop_frac, batch_size, epochs,kernel_size,strides=1,mean_centering=True,optimizer='adam',learning_rate=3*10**-5,momentum=0.9):
		self.neural_net=PLSRCNN.myNeuralNet(number_of_layers, layer_size, drop_frac, batch_size, epochs,kernel_size,strides=strides,
            optimizer=optimizer,learning_rate=learning_rate,momentum=momentum)
		self.scaler=StandardScaler(copy=True, with_mean=mean_centering, with_std=scaling)
	def fit(self, training, truevalues):
		self.scaler.fit(training)
		transformedTraining = self.scaler.transform(training)
		self.neural_net.fit(transformedTraining, truevalues)
	def predict(self, dataset):
		transformedDataset=self.scaler.transform(dataset)
		return np.rot90([self.neural_net.predict(transformedDataset).reshape(-1)], 3)
