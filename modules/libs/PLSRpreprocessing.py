import types
import copy
import numpy as np
import os
import sys
import scipy.signal

def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

def get_buttons():

	buttons=[
	{'key': 'RNNtab0name', 'type': 'tabname', 'text': 'Preprocessing', 'tab': 0} ,

	{'key': 'try_all_normalize', 'type': 'check', 'text': 'Try all', 'tab': 0, 'row': 0} ,
	{'key': 'baseline_value', 'type': 'check', 'text': 'subrtact first variable as baseline', 'tab': 0, 'row': 0} ,
	{'key': 'baseline_linear', 'type': 'check', 'text': 'subtract linear background', 'tab': 0, 'row': 0} ,
	{'key': 'baseline_background', 'type': 'check', 'text': 'subtract reference specta', 'tab': 0, 'row': 0} ,
	{'key': 'select_background_spectra', 'type': 'click', 'text': 'select background spectra', 'bind': select_background_spectra, 'tab': 0, 'row': 0} ,
	{'key': 'background_spectra', 'type': 'txt', 'text': 'background spectra', 'default': '', 'width': 60, 'tab': 0, 'row': 0} ,

	{'key': 'binning', 'type': 'txt:int:range', 'text': 'Dimensionality reduction by binning', 'width': 20, 'default': '1', 'tab': 0, 'row': 2} ,
	#{'key': 'mean_centering', 'type': 'check', 'text': 'Mean centering', 'tab': 0, 'row': 2, 'default': True} ,
	{'key': 'mean_centering', 'type': 'radio:text', 'texts': ['No mean centering', 'Mean centering','Try all'], 'default': '1', 'tab': 0, 'row': 2} ,
#{'key': 'scaling', 'type': 'check', 'text': 'Scaling', 'tab': 0, 'row': 2} ,
	{'key': 'scaling', 'type': 'radio:text', 'texts': ['No scaling', 'Scaling','Try all'], 'tab': 0, 'row': 2} ,


	{'key': 'RegressionL2', 'type': 'label', 'text': 'SG and derivative: ', 'tab': 0, 'row': 3} ,
	#{'key': 'use_SG', 'type': 'check', 'text': 'useSG', 'tab': 0, 'row': 3} ,
	{'key': 'use_SG', 'type': 'radio:text', 'texts': ['No SG', 'use SG', 'Both'], 'tab': 0, 'row': 4} ,

	{'key': 'SG_window_min', 'type': 'txt:int', 'text': 'SG:MinW ', 'default': '9', 'width': 4, 'tab': 0, 'row': 3} ,
	{'key': 'SG_window_max', 'type': 'txt:int', 'text': 'SG:MaxW ', 'default': '11', 'width': 4, 'tab': 0, 'row': 3} ,
	{'key': 'SG_order_min', 'type': 'txt:int', 'text': 'SG Order Min', 'default': '1', 'width': 4, 'tab': 0, 'row': 3} ,
	{'key': 'SG_order_max', 'type': 'txt:int', 'text': 'SG Order Max', 'default': '1', 'width': 4, 'tab': 0, 'row': 3} ,

	{'key': 'derivative', 'type': 'radio:text', 'texts': ['Not der', '1st der', '2nd der', 'all'], 'tab': 0, 'row': 4} ,

	{'key': 'try_all_scatter_correction', 'type': 'check', 'text': 'try all', 'tab': 0, 'row': 5} ,
	{'key': 'normalize', 'type': 'check', 'text': 'Normalize', 'tab': 0, 'row': 5} ,
	{'key': 'SNV_key', 'type': 'check', 'text': 'SNV', 'tab': 0, 'row': 5} ,
	{'key': 'MSC_key', 'type': 'check', 'text': 'MSC', 'tab': 0, 'row': 5} ,
	#{'key': 'EMSC_key', 'type': 'check', 'text': 'EMSC', 'tab': 0, 'row': 5} ,
	{'key': 'select_reference_spectra', 'type': 'click', 'text': 'select reference spectra', 'bind': select_reference_spectra, 'tab': 0, 'row': 5} ,
	{'key': 'reference_spectra', 'type': 'txt', 'text': 'reference spectra', 'default': '', 'width': 60, 'tab': 0, 'row': 5} ,

	{'key': 'filter', 'type': 'radio:text', 'texts': ['No filter', 'MA', 'Butterworth', 'Hamming','Fourier','Try all'], 'tab': 0, 'row': 7} ,
	{'key': 'filterN', 'type': 'txt:int:range', 'default': '2', 'text': 'Butterworth n', 'tab': 0, 'row': 7} ,
	{'key': 'sb','type': 'txt:float:range', 'default': '0.2', 'text': 'Butterworth sb', 'tab': 0, 'row': 7} ,

	#{'key': 'fourier_filter', 'type': 'check', 'text': 'Fourier', 'tab': 0, 'row': 8} ,
	{'key': 'fourier_filter_cut', 'type': 'txt:int:range', 'text': 'cutoff', 'default': '100', 'width': 4, 'tab': 0, 'row': 8} ,
	{'key': 'plot_fourier', 'type': 'check', 'text': 'plot Fourier spectra', 'tab': 0, 'row': 8} ,
	{'key': 'plot_fourier_log', 'type': 'check', 'text': 'log', 'tab': 0, 'row': 8} ,
	{'key': 'fourier_window', 'type': 'radio:text', 'texts': ['None','blackman','blackmanharris', 'hamming', 'hann'], 'tab': 0, 'row': 8} ,
	{'key': 'fourier_window_size_multiplier', 'type': 'txt:float:range', 'text': 'window size multiplier', 'default': '1.1', 'width': 4, 'tab': 0, 'row': 8} ,
	{'key': 'reverse_fourier_window', 'type': 'check', 'text': 'inverse window after', 'default':True, 'tab': 0, 'row': 8} ,



	{'key': 'plot_spectra_before_preprocessing', 'type': 'check', 'text': 'Plot spectra before preprocessing', 'tab': 0, 'row': 9, 'default':True} ,

	{'key': 'plot_spectra_after_preprocessing', 'type': 'check', 'text': 'Plot spectra after preprocessing', 'tab': 0, 'row': 9} ,

	#on tab wavelength selection:
	{'key': 'windows', 'type': 'txt', 'text': 'Data range', 'default': ':,', 'width': 20, 'tab': 2, 'row': 0} ,
	{'key': 'select_active_wavenumers', 'type': 'click', 'text': 'select active wavenumers', 'bind': select_active_wavenumers, 'tab': 2, 'row': 1} ,
	{'key': 'active_wavenumers_file', 'type': 'txt', 'text': 'active wavenumers', 'default': '', 'width': 60, 'tab': 2, 'row': 1} ,
	]
	return buttons


def do_preprocessing(run,T,V):
	ui=run.ui
	#build initial case
	initial_case=types.SimpleNamespace()
	initial_case.T=T
	initial_case.V=V
	initial_case.folder=run.filename+'/'
	initial_case.preprocessing_done=[]
	initial_case.reference_spectra=run.reference_spectra
	initial_case.background_spectra=run.background_spectra
	cases = do_binning(ui,initial_case,run.wavenumbers)
	cases = do_scatter_cor(ui,cases)
	cases = do_filter(ui,cases)
	cases = do_sg_and_der(run,cases)
	cases = do_baseline(ui,cases)
	run.preprocessed_cases = cases
	# do binning

def do_binning(ui,initial_case,wavenumbers):
	binned_cases=[]
	for bin_size in ui['binning']:
		cur_case=copy.deepcopy(initial_case)
		cur_case.T,cur_case.V,cur_case.reference_spectra,cur_case.background_spectra,cur_case.wavenumbers=bin_me(cur_case.T,cur_case.V,cur_case.reference_spectra,cur_case.background_spectra,wavenumbers,bin_size)
		if len(ui['binning'])>1:
			cur_case.folder+='binned_'+str(bin_size)+'/'
		cur_case.preprocessing_done.append('Binned '+str(bin_size))
		binned_cases.append(cur_case)
	return binned_cases


def do_scatter_cor(ui,inp_cases):
	if ui['try_all_scatter_correction']:
		scatter_options=[1,1,1,1]
	else:
		scatter_options=[0,ui['normalize'],ui['SNV_key'],ui['MSC_key']]
	if sum(scatter_options)==0:
		scatter_options=[1,0,0,0]
	scatter_cor_cases=[]
	for inp_case in inp_cases:
		for j,active in enumerate(scatter_options):
			if active:
				cur_case=copy.deepcopy(inp_case)
				if j==0: # 'no scatter cor'
					if np.sum(scatter_options)>1:
						cur_case.folder+='no scatter cor/'

				if j==1:
					for E in [cur_case.T,cur_case.V]:
						for i,_ in enumerate(E.Y):
							E.X[i]=normalizeVec(E.X[i])
					if np.sum(scatter_options)>1:
						cur_case.folder+='normalized/'
					cur_case.preprocessing_done.append('Normalized individual spectra')
				if j==2: # 'SNV_key'
					for E in [cur_case.T,cur_case.V]:
						for i,_ in enumerate(E.Y):
							E.X[i]=snv_fun(E.X[i])
					if np.sum(scatter_options)>1:
						cur_case.folder+='SNV/'
					cur_case.preprocessing_done.append('SNV')
				if j==3: # 'MSC_key'
					cur_case.T.X, MSC_ref_train = msc_fun(cur_case.T.X,cur_case.reference_spectra)
					# if if not reference spectra is set by the user, run.reference_spectra will be None
					# if run.reference_spectra==None, MSC_ref_train will be np.mean(T.X,acis=0)
					# otherwise MSC_ref_train will be run.reference_spectra
					cur_case.V.X, _ = msc_fun(cur_case.V.X, MSC_ref_train) # this will always be validation set, if available
					if np.sum(scatter_options)>1:
						cur_case.folder+='MSC/'
					cur_case.preprocessing_done.append('MSC')
				scatter_cor_cases.append(cur_case)
	return scatter_cor_cases

def do_sg_and_der(run,inp_cases):
	sg_cases=[]
	for cur_case in inp_cases:
		sg_cases+=Sgolay(cur_case,cur_case.wavenumbers,run.ui)
	der_cases=[]
	for sgcase in sg_cases:
		der_cases+=Derivatives(sgcase,run.ui)
	return der_cases

def do_filter(ui,inp_cases):
	if ui['filter']=='Try all':
		filter_options=[1,1,1,1,1]
	else:
		i=ui['filter']
		filter_options=[0,i=='MA',i=='Butterworth',i=='Hamming',i=='Fourier']
	if sum(filter_options)==0:
		filter_options=[1,0,0,0,0]
	filter_cases=[]
	for inp_case in inp_cases:
		for j,active in enumerate(filter_options):
			if active:
				if j==0: # 'no filter'
					cur_case=copy.deepcopy(inp_case)
					if np.sum(filter_options)>1:
						cur_case.folder+='/no filter/'
					filter_cases.append(cur_case)
				if j==1: # 'MA'
					for N in ui['filterN']:
						cur_case=copy.deepcopy(inp_case)
						for E in [cur_case.T,cur_case.V]:
							if len(E.X)>0:
								E.X=MA(N,E.X)
						if np.sum(filter_options)>1:
							cur_case.folder+='MA/'
						cur_case.preprocessing_done.append('MA N = '+str(N))
						filter_cases.append(cur_case)
				if j==2: # 'Butterworth'
					for N in ui['filterN']:
						for sb in ui['sb']:
							cur_case=copy.deepcopy(inp_case)
							for E in [cur_case.T,cur_case.V]:
								if len(E.X)>0:
									E.X=butterworth(N, sb,E.X)
							if np.sum(filter_options)>1:
								cur_case.folder+='Butterworth/'
							cur_case.preprocessing_done.append('Butterworth N = '+str(N)+' sb = '+str(sb))
							filter_cases.append(cur_case)
				if j==3: # 'Hamming'
					for N in ui['filterN']:
						cur_case=copy.deepcopy(inp_case)
						for E in [cur_case.T,cur_case.V]:
							if len(E.X)>0:
								E.X=Hamming(N,E.X)
						if np.sum(filter_options)>1:
							cur_case.folder+='Hamming/'
						cur_case.preprocessing_done.append('Hamming N = '+str(N))
						filter_cases.append(cur_case)
				if j==4: # 'MFourierA'
					for fourier_filter_cut in ui['fourier_filter_cut']:
						for fourier_window_size_multiplier in ui['fourier_window_size_multiplier']:
							cur_case=copy.deepcopy(inp_case)
							for E in [cur_case.T,cur_case.V]:
								if len(E.X)>0:
									fourier_filter(E,ui,fourier_filter_cut,fourier_window_size_multiplier)
							if np.sum(filter_options)>1:
								cur_case.folder+='Fourier/'
							cur_case.preprocessing_done.append('Fourier filter cut = '+str(fourier_filter_cut)+' windown mul = '+str(fourier_window_size_multiplier))
							filter_cases.append(cur_case)
	return filter_cases

def do_baseline(ui,inp_cases):
	if ui['try_all_normalize']:
		normalize_options=[1,1,1,1]
	else:
		normalize_options=[0,ui['baseline_value'],ui['baseline_linear'],ui['baseline_background']]
		#normalize_options=[0,ui['normalize'],ui['baseline_value'],ui['baseline_linear'],ui['baseline_background']]
	if sum(normalize_options)==0:
		normalize_options=[1,0,0,0]

	normalize_cases=[]
	for inp_case in inp_cases:
		for j,active in enumerate(normalize_options):
			if active:
				if j==0:
					cur_case=copy.deepcopy(inp_case)
					if np.sum(normalize_options)>1:
						cur_case.folder+='no_baseline_cor/'
					normalize_cases.append(cur_case)
				'''if j==1:
					cur_case=copy.deepcopy(inp_case)
					for E in [cur_case.T,cur_case.V]:
						for i,_ in enumerate(E.Y):
							E.X[i]=normalizeVec(E.X[i])
					if np.sum(normalize_options)>1:
						cur_case.folder+='normalized/'
					cur_case.preprocessing_done.append('Normalized individual spectra')
					normalize_cases.append(cur_case)'''
				if j==1:
					cur_case=copy.deepcopy(inp_case)
					for E in [cur_case.T,cur_case.V]:
						for i,_ in enumerate(E.Y):
							E.X[i]=baseline_value_corr(E.X[i])
					if np.sum(normalize_options)>1:
						cur_case.folder+='baseline_value/'
					cur_case.preprocessing_done.append('Basline corrected')
					normalize_cases.append(cur_case)
				if j==2:
					cur_case=copy.deepcopy(inp_case)
					for E in [cur_case.T,cur_case.V]:
						for i,_ in enumerate(E.Y):
							E.X[i]=baseline_linear_corr(E.X[i])
					if np.sum(normalize_options)>1:
						cur_case.folder+='baseline_linear/'
					cur_case.preprocessing_done.append('Subtracted linear baseline')
					normalize_cases.append(cur_case)
				if j==3:
					cur_case=copy.deepcopy(inp_case)
					if not hasattr(cur_case.background_spectra,'len'): #background spectra not loaded correctly
						eprint('background spectra not loaded correctly, skipping baseline_background')
						continue
					for E in [cur_case.T,cur_case.V]:
						for i,_ in enumerate(E.Y):
							E.X[i]=baseline_background_corr(E.X[i],cur_case.background_spectra)
					if np.sum(normalize_options)>1:
						cur_case.folder+='baseline_background/'
					cur_case.preprocessing_done.append('Subtracted background spectra')
					normalize_cases.append(cur_case)
	return normalize_cases



def baseline_value_corr(case):
	"""Baseline correction that sets the first independent variable of each
	spectrum to zero."""
	position = 0
	subtract = case[position]
	return (case-subtract)

def baseline_linear_corr(case):
	"""Baseline correction that subtracts a linearly baseline between
	the first and last independent variable."""
	l=len(case)
	dydx=(case[-1]-case[0])/(l-1)
	subtract=np.arange(case[0],case[-1]+dydx*0.5,dydx)
	return case-subtract

def baseline_background_corr(case,bg):
	"""Subtracts refenence"""
	return (case-bg)

def MA(n,inp):
		b = [1.0/n]*n
		a = 1
		return scipy.signal.filtfilt(b,a,inp)

def butterworth(n, sb, inp):
		[b, a] = scipy.signal.butter(n, sb)
		return scipy.signal.filtfilt(b,a,inp)

def normalizeVec(inp):
	""" Normalize intensity of spectra
	"""
	inp=np.array(inp)
	factor=len(inp)/np.sqrt(sum(inp**2))
	return inp*factor

def snv_fun(case):
	"""Scatter correction through standard normal variate.
	"""
	case_snv = np.zeros(len(case))
	case_snv = (case - np.mean(case)) / np.std(case)
	return case_snv

def msc_fun(cases, reference=None):
	"""Function for multiplicative scatter correction, which corrects scatter effects
	based on a reference spectrum. If a reference spectrum is not available, it
	uses the average spectrum based on training data."""

	#First we do mean centre correction
	for i in range(cases.shape[0]):
		cases[i,:] -= cases[i,:].mean()

	#Set reference spectrum
	if reference is None:
		ref = np.mean(cases, axis=0)
	else:
		ref = reference

	cases_msc = np.zeros_like(cases)
	for i in range(cases.shape[0]):
		fit = np.polyfit(ref, cases[i,:], 1, full=True)
		cases_msc[i,:] = (cases[i,:] - fit[0][1]) / fit[0][0]

	return (cases_msc, ref)


def Hamming(n,inp):
		b = scipy.signal.firwin(n, cutoff = 0.2, window = "hamming")
		a = 1
		return scipy.signal.filtfilt(b,a,inp)

def bin_me(T,V,ref_spectra,bg_spectra,wavenumbers,bin_size):
	if bin_size==1:
		return T,V,ref_spectra,bg_spectra,wavenumbers
	else:
		old_len=len(wavenumbers)
		new_len=old_len//bin_size
		print('binning data in groups of '+str(bin_size))
		print(str(old_len)+' datapoints become '+str(new_len))
		if not new_len*bin_size==old_len:
			print('dropped last '+str(old_len-new_len*bin_size)+' datapoints')
		wavenumbers = bin_vector(wavenumbers, bin_size)
		temp=T.X
		T.X=np.zeros((T.X.shape[0],len(wavenumbers)))
		for i , _ in enumerate(T.X):
			T.X[i] = bin_vector(temp[i],bin_size)
		temp=V.X
		V.X=np.zeros((V.X.shape[0],len(wavenumbers)))
		for i , _ in enumerate(V.X):
			V.X[i] = bin_vector(temp[i],bin_size)
		if not ref_spectra is None:
			ref_spectra=bin_vector(ref_spectra,bin_size)
		if not bg_spectra is None:
			bg_spectra=bin_vector(bg_spectra,bin_size)
	return T,V,ref_spectra,bg_spectra,wavenumbers

def bin_vector(data, bin_size):
	return data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size).mean(axis=1)




def Sgolay(case,wavenumbers,ui): #sgolay for using Moving Window
	sg_cases=[]
	if ui['use_SG']=='use SG' or ui['use_SG']=='Both':# 'use_SG', 'type': 'radio:text', 'texts': ['No SG', 'use SG', 'Both']
		if ui['derivative']=='Not der':
			ders=[0]
		elif ui['derivative']=='1st der':
			ders=[1]
		elif ui['derivative']=='2nd der':
			ders=[2]
		elif ui['derivative']=='all':
			ders=[0,1,2]
		for filtersize in range(ui['SG_window_min'],ui['SG_window_max']+1,2):
			for order in range(ui['SG_order_min'],ui['SG_order_max']+1):
				for der in ders:
					if der>order:
						continue
					sg_cases.append(copy.deepcopy(case))
					folder=case.folder
					sg_config=types.SimpleNamespace()
					T2=types.SimpleNamespace()
					T2.Y=sg_cases[-1].T.Y
					V2=types.SimpleNamespace()
					V2.Y=sg_cases[-1].V.Y
					#create folder for storing results
					folderSG=folder+'SG_size'+str(filtersize)+'_order'+str(order)+'/'
					if der==0:
						if ui['derivative']=='Not der':
							folderSG=folderSG #do not include derivative information in folder
						elif ui['derivative']=='all':
							folderSG=folderSG+'NotDer/'
					elif der==1:
						folderSG=folderSG+'1stDer/'
						sg_cases[-1].preprocessing_done.append('First Derivative')
					elif der==2:
						folderSG=folderSG+'2ndDer/'
						sg_cases[-1].preprocessing_done.append('Second Derivative')
					sg_cases[-1].preprocessing_done.append('SGFilterSize: '+str(filtersize))
					sg_cases[-1].preprocessing_done.append('SGOrder: '+str(order))
					if ui['save_check_var'] and not os.path.exists(folderSG) and not ui['do_not_save_plots']:
						os.makedirs(folderSG)
					#sgolay for training set
					TXSG=[]
					for i in range(len(sg_cases[-1].T.X)):
						X = scipy.signal.savgol_filter(sg_cases[-1].T.X[i], filtersize, order, der)
						TXSG.append(X)
					T2.X=np.array(TXSG)
					#sgolay for validation set
					VXSG=[]
					if ui['is_validation']=='Training and Validation':
						for i in range(len(sg_cases[-1].V.X)):
							X = scipy.signal.savgol_filter(sg_cases[-1].V.X[i], filtersize, order, der)
							VXSG.append(X)
							V2.X=np.array(VXSG)
					else:
						V2=sg_cases[-1].V
					sg_config.derivative=der
					sg_config.curSGOrder=order
					sg_config.curSGFiltersize=filtersize
					sg_cases[-1].T=T2
					sg_cases[-1].V=V2
					sg_cases[-1].wavenumbers=wavenumbers
					sg_cases[-1].folder=folderSG
					sg_cases[-1].sg_config=sg_config
					sg_cases[-1].used_sg=True
	if ui['use_SG']=='No SG' or ui['use_SG']=='Both':# 'use_SG', 'type': 'radio:text', 'texts': ['No SG', 'use SG', 'Both']
		sg_cases.append(copy.deepcopy(case))
		if ui['save_check_var'] and not os.path.exists(case.folder)  and not ui['do_not_save_plots']:
			os.makedirs(case.folder)
		sg_config=types.SimpleNamespace()
		sg_config.derivative=0
		sg_config.curSGOrder=None
		sg_config.curSGFiltersize=None
		sg_cases[-1].wavenumbers=wavenumbers
		sg_cases[-1].sg_config=sg_config
		sg_cases[-1].used_sg=False
	return sg_cases




def Derivatives(sgcase,ui):  #derivative for using Moving Window
	#traX=T.X
	#traY=T.Y
	#valX=V.X
	#valY=V.Y
	der_cases=[]
	# not derivative
	if ui['derivative']=='Not der' or ui['derivative']=='all' or sgcase.used_sg==True:
		#create sgcase.folder for storing results
		if sgcase.used_sg==False:
			if ui['derivative']=='all': # if not derrivative, do not make a new sgcase.folder
				folderNotDer=sgcase.folder+'NotDer/'
				if not os.path.exists(folderNotDer) and ui['save_check_var'] and not ui['do_not_save_plots']:
					os.makedirs(folderNotDer)
			elif ui['derivative']=='Not der':
				folderNotDer=sgcase.folder
		else:
			folderNotDer=sgcase.folder
		#make case
		der_cases.append(copy.deepcopy(sgcase))
		der_cases[-1].derrivative=sgcase.sg_config.derivative
		der_cases[-1].folder=folderNotDer
		#do MW
		#der_cases.append([sgcase.T,sgcase.V,sgcase.wavenumbers, folderNotDer,ui,sgcase.sg_config,curDerivative+sgcase.sg_config.derivative])
		#MW(T,V,sgcase.wavenumbers,ui, folderNotDer+'/',ui)
	# First Der
	if sgcase.used_sg==False:
		if ui['derivative']=='1st der' or ui['derivative']=='all':
			curDerivative=1
			T1=types.SimpleNamespace()
			T1.Y=sgcase.T.Y
			V1=types.SimpleNamespace()
			V1.Y=sgcase.V.Y
			#create sgcase.folder for storing results
			folderFirstDer=sgcase.folder+'1stDer/'
			if not os.path.exists(folderFirstDer) and ui['save_check_var'] and not ui['do_not_save_plots']:
				os.makedirs(folderFirstDer)
			#differentiate training
			TXDer=[]
			for i in range(len(sgcase.T.X)):
				dwave,X = Der(sgcase.wavenumbers,sgcase.T.X[i])
				TXDer.append(X)
			T1.X=np.array(TXDer)
			#differentiate validation
			if ui['is_validation']=='Training and Validation':
				VXDer=[]
				for i in range(len(sgcase.V.X)):
					dwave,X = Der(sgcase.wavenumbers,sgcase.V.X[i])
					VXDer.append(X)
				V1.X=np.array(VXDer)
			else:
				V1.X=sgcase.V.X
			#make case
			der_cases.append(copy.deepcopy(sgcase))
			der_cases[-1].T=T1
			der_cases[-1].V=V1
			der_cases[-1].wavenumbers=dwave
			der_cases[-1].derrivative=curDerivative
			der_cases[-1].folder=folderFirstDer
			der_cases[-1].preprocessing_done.append('First Derivative')
			#MW(T1,V1,dwave,ui, folderFirstDer+'/',ui)
		# second der
		if ui['derivative']=='2nd der' or ui['derivative']=='all':
			curDerivative=2
			T2=types.SimpleNamespace()
			T2.Y=sgcase.T.Y
			V2=types.SimpleNamespace()
			V2.Y=sgcase.V.Y
			#create sgcase.folder for storing results
			folderSecondDer=sgcase.folder+'2ndDer/'
			if not os.path.exists(folderSecondDer) and ui['save_check_var'] and not ui['do_not_save_plots']:
				os.makedirs(folderSecondDer)
			#differentiate training
			TX2Der=[]
			for i in range(len(sgcase.T.X)):
				dwave,X = Der2(sgcase.wavenumbers,sgcase.T.X[i])
				TX2Der.append(X)
			T2.X=np.array(TX2Der)
			#differentiate validation
			if ui['is_validation']=='Training and Validation':
				VX2Der=[]
				for i in range(len(sgcase.V.X)):
					dwave,X = Der2(sgcase.wavenumbers,sgcase.V.X[i])
					VX2Der.append(X)
				V2.X=np.array(VX2Der)
			else:
				V2.X=sgcase.V.X
			#do MW
			#make case
			der_cases.append(copy.deepcopy(sgcase))
			der_cases[-1].T=T2
			der_cases[-1].V=V2
			der_cases[-1].wavenumbers=dwave
			der_cases[-1].derrivative=curDerivative
			der_cases[-1].folder=folderSecondDer
			der_cases[-1].preprocessing_done.append('Second Derivative')
			#MW(T2,V2,dwave,ui, folderSecondDer+'/',ui)
	return der_cases


def Der(x,y):
	"""Function for finding first derivative of spectral data. Uses finite differences."""
	n=len(x)
	x2=np.zeros(n-1)
	y2=np.zeros(n-1)
	for i in range(n-1):
		x2[i]=0.5*(x[i]+x[i+1])
		y2[i]=(y[i+1]-y[i])/(x[i+1]-x[i])
	return(x2,y2)

def Der2(x,y):
	"""Function for finding second derivative of spectral data. Uses finite differences."""
	n=len(x)
	x2=np.zeros(n-2)
	y2=np.zeros(n-2)
	dx2=(x[1]-x[0])**2 # assumed constant
	for i in range(n-2):
		x2[i]=x[i+1]
		y2[i]=(y[i]-2*y[i+1]+y[i+2])/dx2
	return(x2,y2)

def GetDatapoints(wavenumbers, ui):
	reversed=0
	if wavenumbers[0]>wavenumbers[-1]:
		reversed=1
	datapoints=[]
	#datapointlists=[]
	for s in ui['windows'].split(','):
		if not s.strip()=='':
			start=s.split(':')[1].strip()
			stop=s.split(':')[0].strip()
			if not start=='' and not stop=='':
				if float(start)>float(stop):
					start,stop=stop,start
			if reversed:
				start,stop=stop,start
			if start=='':
				start=0
			else:
				st=float(start)
				for i in range(len(wavenumbers)):
					if wavenumbers[i]==st:
						start=i
						break
					elif reversed==1 and wavenumbers[i]<st:
						start=i-1
						break
					elif reversed==0 and wavenumbers[i]>st:
						start=i-1
						break
				if start<0:
					start=0
			if stop=='':
				stop=len(wavenumbers)
			else:
				st=float(stop)
				for i in range(len(wavenumbers)):
					if wavenumbers[i]==st:
						stop=i+1
						break
					elif reversed==1 and wavenumbers[i]<st:
						stop=i+1
						break
					elif reversed==0 and wavenumbers[i]>st:
						stop=i+1
						break
				if i==len(wavenumbers)-1:
					stop=len(wavenumbers)
			r=list(range(start,stop))
			if not start==0 or not stop==len(wavenumbers):
				print('range '+str(round(wavenumbers[start],1))+' to '+str(round(wavenumbers[stop-1],1))+' datapoints: '+str(len(r)))
			datapoints=datapoints+r
			#datapointlists.append(r)
	active_wavenumers=np.zeros(len(wavenumbers), dtype=bool)
	active_wavenumers[datapoints]=True
	if not ui['active_wavenumers_file'] =='':
		# read the active wavenumbers file
		file_active_wavenumers=[]
		with open(ui['active_wavenumers_file']) as f:
			for line in f:
				if line.split('\t')[1].strip()=='True':
					file_active_wavenumers.append(float(line.split('\t')[0].strip()))
					print(file_active_wavenumers[-1])
		file_active_wavenumers=np.array(file_active_wavenumers)
		#remove wavenumbers not found in active wavenumbers file
		## we allow leniency of 0.6 dwavenum so that we can find wavenumbers based on the derivative, and apply them to a spectra where we have not taken the derrivative
		dwavenum=wavenumbers[1]-wavenumbers[0]
		for i, wavenum in enumerate(wavenumbers):
			if active_wavenumers[i]==True:
				closest_wavenumber= min(file_active_wavenumers, key=lambda x:abs(x-wavenum))
				if abs(wavenum-closest_wavenumber)>0.6*abs(dwavenum):
					active_wavenumers[i]=False
	#for i,j in zip(wavenumbers,active_wavenumers):
	#	print(i,j)
	#generate datapointlists from datapoints
	return active_wavenumers

def fourier_filter(E,ui,fourier_filter_cut,fourier_window_size_multiplier):
	#{'key': 'fourier_window', 'type': 'radio:text', 'texts': ['None','blackman','blackmanharris', 'hamming', 'hann'], 'tab': 0, 'row': 7} ,
	factor=fourier_window_size_multiplier#ui['fourier_window_size_multiplier']
	win_size=int(E.X.shape[1]*factor)
	if ui['fourier_window']=='None': window=np.ones(win_size)
	if ui['fourier_window']=='blackman': window=scipy.signal.blackman(win_size,sym=False)
	if ui['fourier_window']=='blackmanharris': window=scipy.signal.blackmanharris(win_size,sym=False)
	if ui['fourier_window']=='hamming': window=scipy.signal.hamming(win_size,sym=False)
	if ui['fourier_window']=='hann': window=scipy.signal.hann(win_size,sym=False)
	#window=np.ones(E.X.shape[1])
	start=(win_size-E.X.shape[1])//2
	window=window[start:start+E.X.shape[1]]
	E.X_fft=scipy.fftpack.rfft(E.X*window)
	E.X_fft_uncut=copy.deepcopy(E.X_fft)
	n=fourier_filter_cut#ui['fourier_filter_cut']

	xax=np.arange(E.X_fft.shape[1])
	for i,x in enumerate(xax):
		if abs(x)>n:
			E.X_fft[:,i]=0
	if ui['reverse_fourier_window']:
		window[window<10**-3]=1
		E.X=scipy.fftpack.irfft(E.X_fft).real/window
	else:
		E.X=scipy.fftpack.irfft(E.X_fft).real

import tkinter.filedialog
def select_reference_spectra(event):
	#global run
	filename = tkinter.filedialog.askopenfilename() # show an "Open" dialog box and return the path to the selected file
	event.widget.master.master.master.master.master.buttons['reference_spectra'].set(filename)

def select_background_spectra(event):
	#global run
	filename = tkinter.filedialog.askopenfilename() # show an "Open" dialog box and return the path to the selected file
	event.widget.master.master.master.master.master.buttons['background_spectra'].set(filename)

def select_active_wavenumers(event):
	#global run
	filename = tkinter.filedialog.askopenfilename() # show an "Open" dialog box and return the path to the selected file
	event.widget.master.master.master.master.master.buttons['active_wavenumers_file'].set(filename)
