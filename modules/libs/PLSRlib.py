
import numpy as np
import scipy

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

def mlr(x,y,order):
	"""Multiple linear regression fit of the columns of matrix x
	(dependent variables) to constituent vector y (independent variables)
	order -     order of a smoothing polynomial, which can be included
	in the set of independent variables. If order is
	not specified, no background will be included.
	b -         fit coeffs
	f -         fit result (m x 1 column vector)
	r -         residual   (m x 1 column vector)
	"""
	if order > 0:
		s=scipy.ones((len(y),1))
		for j in range(order):
			s=scipy.concatenate((s,(scipy.arange(0,1+(1.0/(len(y)-1))-0.5/(len(y)-1),1.0/(len(y)-1))**j)[:,nA]),1)
		X=scipy.concatenate((x, s),1)
	else:
		X = x
	b = scipy.dot(scipy.dot(scipy.linalg.pinv(scipy.dot(scipy.transpose(X),X)),scipy.transpose(X)),y)
	f = scipy.dot(X,b)
	r = y - f
	return b,f,r

def emsc(case, order, fit=None):
	"""Extended multiplicative scatter correction
	case -   spectral data for background correction
	order -     order of polynomial
	fit -       if None then use average spectrum, otherwise provide a spectrum
				as a column vector to which all others fitted
	corr -      EMSC corrected data
	mx -        fitting spectrum
	"""
	if not type(fit)==type(None):
		mx = fit
	else:
		mx = scipy.mean(case,axis=0)[:,nA]
	corr = scipy.zeros(case.shape)
	for i in range(len(case)):
		b,f,r = mlr(mx, case[i,:][:,nA], order)
		corr[i,:] = scipy.reshape((r/b[0,0]) + mx, (corr.shape[1],))
	corr=np.nan_to_num(corr)
	return corr

def baseline_corr(case):
	"""Baseline correction that sets the first independent variable of each
	spectrum to zero."""
	size = case.shape
	subtract = scipy.transpose(scipy.resize(scipy.transpose(case[:,0]),(size[1],size[0])))
	return (case-subtract)

def baseline_avg(case):
	"""Baseline correction that subtracts an average of the first and last
	independent variable from each variable."""
	size = case.shape
	subtract = scipy.transpose(scipy.resize(scipy.transpose((case[:,0]+case[:size[1]-1])/2),(size[1],size[0])))
	return (case-subtract)

def baseline_linear(case):
	"""Baseline correction that subtracts a linearly increasing baseline between
	the first and last independent variable."""
	size, t = case.shape, 0
	subtract = scipy.zeros((size[0],size[1]), 'd')
	while t < size[0]:
		a = case[t,0]
		b = case[t,size[1]-1]
		div = (b-a)/size[1]
		if div == 0:
			div = 1
		arr = scipy.arrange(a,b,div,'d')
		subtract[t,:] = scipy.resize(arr,(size[1],))
		t = t+1
	return case-subtract
