import types
import copy
import numpy as np
import os

import scipy.signal
def get_buttons():

    buttons=[
    {'key': 'RNNtab0name', 'type': 'tabname', 'text': 'Preprocessing', 'tab': 0} ,

    {'key': 'normalize', 'type': 'check', 'text': 'Normalize individual spectra before preprocessing', 'tab': 0, 'row': 0} ,
    {'key': 'baseline_value', 'type': 'check', 'text': 'subrtact first variable as baseline', 'tab': 0, 'row': 0} ,
    {'key': 'baseline_linear', 'type': 'check', 'text': 'subtract linear background', 'tab': 0, 'row': 0} ,
    {'key': 'baseline_background', 'type': 'check', 'text': 'subtract reference specta', 'tab': 0, 'row': 0} ,
    {'key': 'select_background_spectra', 'type': 'click', 'text': 'select background spectra', 'bind': select_background_spectra, 'tab': 0, 'row': 0} ,
    {'key': 'background_spectra', 'type': 'txt', 'text': 'background spectra', 'default': '', 'width': 60, 'tab': 0, 'row': 0} ,

    {'key': 'binning', 'type': 'txt:int', 'text': 'Dimensionality reduction by binning', 'width': 4, 'default': '1', 'tab': 0, 'row': 2} ,
    {'key': 'mean_centering', 'type': 'check', 'text': 'Mean centering', 'tab': 0, 'row': 2, 'default': True} ,
    {'key': 'scaling', 'type': 'check', 'text': 'Scaling', 'tab': 0, 'row': 2} ,

    {'key': 'RegressionL2', 'type': 'label', 'text': 'SG and derivative: ', 'tab': 0, 'row': 3} ,
    {'key': 'use_SG', 'type': 'check', 'text': 'useSG', 'tab': 0, 'row': 3} ,
    {'key': 'SG_window_min', 'type': 'txt:int', 'text': 'SG:MinW ', 'default': '9', 'width': 4, 'tab': 0, 'row': 3} ,
    {'key': 'SG_window_max', 'type': 'txt:int', 'text': 'SG:MaxW ', 'default': '11', 'width': 4, 'tab': 0, 'row': 3} ,
    {'key': 'SG_order_min', 'type': 'txt:int', 'text': 'SG Order Min', 'default': '1', 'width': 4, 'tab': 0, 'row': 3} ,
    {'key': 'SG_order_max', 'type': 'txt:int', 'text': 'SG Order Max', 'default': '1', 'width': 4, 'tab': 0, 'row': 3} ,

    {'key': 'derivative', 'type': 'radio:text', 'texts': ['Not der', '1st der', '2nd der', 'all'], 'tab': 0, 'row': 4} ,

    {'key': 'SNV_key', 'type': 'check', 'text': 'SNV', 'tab': 0, 'row': 5} ,
    {'key': 'MSC_key', 'type': 'check', 'text': 'MSC', 'tab': 0, 'row': 5} ,
    {'key': 'EMSC_key', 'type': 'check', 'text': 'EMSC', 'tab': 0, 'row': 5} ,
    {'key': 'select_reference_spectra', 'type': 'click', 'text': 'select reference spectra', 'bind': select_reference_spectra, 'tab': 0, 'row': 5} ,
    {'key': 'reference_spectra', 'type': 'txt', 'text': 'reference spectra', 'default': '', 'width': 60, 'tab': 0, 'row': 5} ,

    {'key': 'filter', 'type': 'radio:text', 'texts': ['No filter', 'MA', 'Butterworth', 'Hamming',], 'tab': 0, 'row': 7} ,
    {'key': 'filterN', 'type': 'txt:int', 'default': '2', 'text': 'Butterworth n', 'tab': 0, 'row': 7} ,
    {'key': 'sb','type': 'txt:float', 'default': '0.2', 'text': 'Butterworth sb', 'tab': 0, 'row': 7} ,

    {'key': 'fourier_filter', 'type': 'check', 'text': 'Fourier', 'tab': 0, 'row': 8} ,
    {'key': 'fourier_filter_cut', 'type': 'txt:int', 'text': 'cutoff', 'default': '100', 'width': 4, 'tab': 0, 'row': 8} ,
    {'key': 'plot_fourier', 'type': 'check', 'text': 'plot Fourier spectra', 'tab': 0, 'row': 8} ,
    {'key': 'plot_fourier_log', 'type': 'check', 'text': 'log', 'tab': 0, 'row': 8} ,
    {'key': 'fourier_window', 'type': 'radio:text', 'texts': ['None','blackman','blackmanharris', 'hamming', 'hann'], 'tab': 0, 'row': 8} ,
    {'key': 'fourier_window_size_multiplier', 'type': 'txt:float', 'text': 'window size multiplier', 'default': '1.1', 'width': 4, 'tab': 0, 'row': 8} ,
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
    datasets=[T]
    if ui['is_validation']=='Training and Validation':
        datasets.append(V)
    for E in datasets:
        for i,_ in enumerate(E.Y):
            if ui['normalize']:
                E.X[i]=normalizeVec(E.X[i])
            if ui['baseline_value']:
                E.X[i]=baseline_value_corr(E.X[i])
            if ui['baseline_linear']:
                E.X[i]=baseline_linear_corr(E.X[i])
            if ui['baseline_background']:
                E.X[i]=baseline_background_corr(E.X[i],run.background_spectra)
        if ui['filter'] == 'MA':
            E.XX=copy.deepcopy(E.X)
            E.X=MA(ui['filterN'],E.XX)
        if ui['filter'] == 'Butterworth':
            E.XX=copy.deepcopy(E.X)
            E.X=butterworth(ui['filterN'], ui['sb'],E.XX)
        if ui['filter'] == 'Hamming':
            E.XX=copy.deepcopy(E.X)
            E.X=Hamming(ui['filterN'],E.XX)
        if ui['fourier_filter']:
            fourier_filter(E,ui)
        for i,_ in enumerate(E.Y):
            if ui['SNV_key']:
                E.X[i]=snv_fun(E.X[i])
        if ui['MSC_key']:
            if not hasattr(run,'MSC_ref_train'):
                T.X, run.MSC_ref_train = msc_fun(T.X,run.reference_spectra) # first time will always be training set
            else:
                V.X, _ = msc_fun(V.X, run.MSC_ref_train) # this will always be validation set, if available


    ################################################################################################
    ############################## Fit data using PCR or PLSR ######################################
    ################################################################################################
    # do binning
    T,V,run.wavenumbers=bin_me(T,V,run.wavenumbers,ui)

    #run.common_variables.original_datapoints=run.common_variables.datapoints
    #run.common_variables.original_datapointlists=run.common_variables.datapointlists
    #run.common_variables.datapoints, run.common_variables.datapointlists=GetDatapoints(run.wavenumbers, ui)
    if ui['use_SG']: #if useSG, don't do regular derivation, do SG derivation instead
        ui['SGderivative']=ui['derivative']
        ui['derivative']='Not der'
    else:
        ui['SGderivative']='Not der'
    run.sg_cases=Sgolay(T,V,run.wavenumbers,ui, run.filename+'/')
    run.preprocessed_cases=[]
    for sgcase in run.sg_cases:
        run.preprocessed_cases+=Derivatives(sgcase,ui)



def normalizeVec(inp):
	inp=np.array(inp)
	factor=len(inp)/np.sqrt(sum(inp**2))
	return inp*factor

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

def snv_fun(case):
	"""Scatter correction through standard normal variate."""
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

def bin_me(T,V,wavenumbers,ui):
	bin_size=ui['binning']
	if bin_size==1:
		return T,V,wavenumbers
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
	return T,V,wavenumbers

def bin_vector(data, bin_size):
	return data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size).mean(axis=1)




def Sgolay(T,V,wavenumbers,ui, folder): #sgolay for using Moving Window
	sg_cases=[]
	if ui['use_SG']:
		if ui['SGderivative']=='Not der':
			ders=[0]
		elif ui['SGderivative']=='1st der':
			ders=[1]
		elif ui['SGderivative']=='2nd der':
			ders=[2]
		elif ui['SGderivative']=='all':
			ders=[0,1,2]
		for filtersize in range(ui['SG_window_min'],ui['SG_window_max']+1,2):
			for order in range(ui['SG_order_min'],ui['SG_order_max']+1):
				for der in ders:
					if der>order:
						continue
					sg_cases.append(types.SimpleNamespace())
					sg_config=types.SimpleNamespace()
					T2=types.SimpleNamespace()
					T2.Y=T.Y
					V2=types.SimpleNamespace()
					V2.Y=V.Y
					#create folder for storing results
					folderSG=folder+'SG_size'+str(filtersize)+'_order'+str(order)+'/'
					if der==0:
						if ui['SGderivative']=='Not der':
							folderSG=folderSG #do not include derivative information in folder
						elif ui['SGderivative']=='all':
							folderSG=folderSG+'NotDer/'
					elif der==1:
						folderSG=folderSG+'1stDer/'
					elif der==2:
						folderSG=folderSG+'2ndDer/'
					if ui['save_check_var'] and not os.path.exists(folderSG):
						os.makedirs(folderSG)
					#sgolay for training set
					TXSG=[]
					for i in range(len(T.X)):
						X = scipy.signal.savgol_filter(T.X[i], filtersize, order, der)
						TXSG.append(X)
					T2.X=np.array(TXSG)
					#sgolay for validation set
					VXSG=[]
					if ui['is_validation']=='Training and Validation':
						for i in range(len(V.X)):
							X = scipy.signal.savgol_filter(V.X[i], filtersize, order, der)
							VXSG.append(X)
							V2.X=np.array(VXSG)
					else:
						V2=V
					sg_config.derivative=der
					sg_config.curSGOrder=order
					sg_config.curSGFiltersize=filtersize
					sg_cases[-1].T=T2
					sg_cases[-1].V=V2
					sg_cases[-1].wavenumbers=wavenumbers
					sg_cases[-1].folder=folderSG
					sg_cases[-1].sg_config=sg_config
	else:
		sg_cases.append(types.SimpleNamespace())
		sg_config=types.SimpleNamespace()
		sg_config.derivative=0
		sg_config.curSGOrder=None
		sg_config.curSGFiltersize=None
		sg_cases[-1].T=T
		sg_cases[-1].V=V
		sg_cases[-1].wavenumbers=wavenumbers
		sg_cases[-1].folder=folder
		sg_cases[-1].sg_config=sg_config
	return sg_cases




def Derivatives(sgcase,ui):  #derivative for using Moving Window
	#traX=T.X
	#traY=T.Y
	#valX=V.X
	#valY=V.Y
	preprocessed_cases=[]
	# not derivative
	if ui['derivative']=='Not der' or ui['derivative']=='all':
		#create sgcase.folder for storing results
		if ui['derivative']=='all': # if not derrivative, do not make a new sgcase.folder
			folderNotDer=sgcase.folder+'NotDer/'
			if not os.path.exists(folderNotDer) and ui['save_check_var']:
				os.makedirs(folderNotDer)
		elif ui['derivative']=='Not der':
			folderNotDer=sgcase.folder
		#make case
		preprocessed_cases.append(types.SimpleNamespace())
		preprocessed_cases[-1].T=copy.copy(sgcase.T)
		preprocessed_cases[-1].V=copy.copy(sgcase.V)
		preprocessed_cases[-1].wavenumbers=sgcase.wavenumbers
		preprocessed_cases[-1].sg_config=sgcase.sg_config
		preprocessed_cases[-1].derrivative=sgcase.sg_config.derivative
		preprocessed_cases[-1].folder=folderNotDer
		#do MW
		#preprocessed_cases.append([sgcase.T,sgcase.V,sgcase.wavenumbers, folderNotDer,ui,sgcase.sg_config,curDerivative+sgcase.sg_config.derivative])
		#MW(T,V,sgcase.wavenumbers,ui, folderNotDer+'/',ui)
	# First Der
	if ui['derivative']=='1st der' or ui['derivative']=='all':
		curDerivative=1
		T1=types.SimpleNamespace()
		T1.Y=sgcase.T.Y
		V1=types.SimpleNamespace()
		V1.Y=sgcase.V.Y
		#create sgcase.folder for storing results
		folderFirstDer=sgcase.folder+'1stDer/'
		if not os.path.exists(folderFirstDer) and ui['save_check_var']:
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
		preprocessed_cases.append(types.SimpleNamespace())
		preprocessed_cases[-1].T=T1
		preprocessed_cases[-1].V=V1
		preprocessed_cases[-1].wavenumbers=dwave
		preprocessed_cases[-1].sg_config=sgcase.sg_config
		preprocessed_cases[-1].derrivative=curDerivative
		preprocessed_cases[-1].folder=folderFirstDer
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
		if not os.path.exists(folderSecondDer) and ui['save_check_var']:
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
		preprocessed_cases.append(types.SimpleNamespace())
		preprocessed_cases[-1].T=T2
		preprocessed_cases[-1].V=V2
		preprocessed_cases[-1].wavenumbers=dwave
		preprocessed_cases[-1].sg_config=sgcase.sg_config
		preprocessed_cases[-1].derrivative=curDerivative
		preprocessed_cases[-1].folder=folderSecondDer
		#MW(T2,V2,dwave,ui, folderSecondDer+'/',ui)
	return preprocessed_cases


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
    #    print(i,j)
    #generate datapointlists from datapoints
    return active_wavenumers

def fourier_filter(E,ui):

    #{'key': 'fourier_window', 'type': 'radio:text', 'texts': ['None','blackman','blackmanharris', 'hamming', 'hann'], 'tab': 0, 'row': 7} ,
    factor=ui['fourier_window_size_multiplier']
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
    n=ui['fourier_filter_cut']

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
