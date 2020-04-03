import numpy as np
import fns
from . import PLSRregressionMethods
from . import PLSRsave
import tkinter
import copy
import sklearn.model_selection
import types
from . import PLSRclassifiers

def get_buttons():
	buttons=[
	{'key': 'RNNtab2name', 'type': 'tabname', 'text': 'Wavelength Selection', 'tab': 2} ,


	{'key': 'RegressionL3', 'type': 'label', 'text': 'Type of wavelength selection:', 'tab': 2, 'row': 2} ,
	{'key': 'regression_wavelength_selection', 'type': 'radio:vertical:text', 'texts': ['No wavelength selection', 'Moving Window', 'Genetic Algorithm','Sequential Feature Selector'], 'tab': 2, 'row': 3} ,

	{'key': 'moving_window_min', 'type': 'txt:float', 'text': 'Min window', 'default': '30', 'width': 4, 'tab': 2, 'row': 4} ,
	{'key': 'moving_window_max', 'type': 'txt:float', 'text': 'Max window', 'default': '100', 'width': 4, 'tab': 2, 'row': 4} ,

	{'key': 'RegressionL4', 'type': 'label', 'text': 'GA options ', 'tab': 2, 'row': 5} ,
	{'key': 'GA_number_of_individuals', 'type': 'txt:int', 'text': 'GA num. Individuals', 'default': '100', 'width': 4, 'tab': 2, 'row': 5} ,
	{'key': 'GA_crossover_rate', 'type': 'txt:float', 'text': 'GA crossover rate', 'default': '0.8', 'width': 4, 'tab': 2, 'row': 5} ,
	{'key': 'GA_mutation_rate', 'type': 'txt:float', 'text': 'GA mutation rate', 'default': '0.001', 'width': 6, 'tab': 2, 'row': 5} ,
	{'key': 'GA_max_number_of_generations', 'type': 'txt:int', 'text': 'GA generations', 'default': '20', 'width': 3, 'tab': 2, 'row': 5} ,

	{'key': 'SFS type', 'type': 'radio:text', 'texts': ['Forward', 'Backward'], 'tab': 2, 'row': 6} ,

	{'key': 'SFS_floating', 'type': 'check', 'text': 'Floating', 'tab': 2, 'row': 6} ,
	{'key': 'SFS_num_after_min', 'type': 'txt:int', 'text': 'Iterations after min', 'default': '30', 'width': 4, 'tab': 2, 'row': 6 },
	{'key': 'SFS_target', 'type': 'txt:int', 'text': 'Target number', 'default': '20', 'width': 4, 'tab': 2, 'row': 6 },
	{'key': 'SFS_max_iterations', 'type': 'txt:int', 'text': 'Max iterations', 'default': '300', 'width': 4, 'tab': 2, 'row': 6 },

	{'key': 'WS_loss_type', 'type': 'radio:text', 'texts': ['X-validation on training', 'RMSEC on training', 'RMSEP on validation'], 'tab': 2, 'row': 8} ,
	{'key': 'WS_cross_val_N', 'type': 'txt:int', 'text': 'WS cross val fold', 'default': '1', 'width': 4, 'tab': 2, 'row': 9} ,
	{'key': 'WS_cross_val_max_cases', 'type': 'txt:int', 'text': 'WS cross val num cases', 'default': '-1', 'width': 4, 'tab': 2, 'row': 9} ,
	]
	return buttons


def MW(case,ui,common_variables,keywords={}):
	T=case.T
	V=case.V
	wavenumbers=case.wavenumbers
	folder=case.folder
	try:
		keywords=case.keywords
	except:
		keywords={}
	WS_getCrossvalSplits([0,1],T,V,ui,use_stored=False)
	# get regression module
	reg_module=PLSRregressionMethods.getRegModule(ui['reg_type'],keywords)
	# Set what datapoints to include, the parameter 'wavenum' is in units cm^-1
	if ui['save_check_var']:
		common_variables.tempax.fig=common_variables.tempfig
	#len_wavenumbers=len(wavenumbers)
	dw=wavenumbers[0]-wavenumbers[1]
	#	Windowsize is input in cm^-1, transform to indexes
	MWmax=int(round(ui['moving_window_max']/abs(dw),0))
	MWmin=int(round(ui['moving_window_min']/abs(dw),0))

	Wresults=np.zeros((len(wavenumbers),MWmax+1-MWmin))
	Wsizes=np.arange(MWmin,MWmax+1)
	# do moving window
	for i,Wsize in enumerate(Wsizes):
		trail_active_wavenumbers=[]
		for j, Wcenter in enumerate(wavenumbers):
			Wstart=j-Wsize//2
			Wend=Wstart+Wsize
			#if Wsize < MWmax+1 and i < len(wavenumbers)+1:
			if Wstart<0:
				k=j
				continue
			elif Wend>len(wavenumbers):
				l=j
				break
			else:
				trail_active_wavenumbers.append(np.arange(Wstart,Wend))
				#Wresults[j,i]=WS_getRMSEP(reg_module,trail_active_wavenumbers[-1],T,V,use_stored=False)
		print('moving window row '+str(i)+' of  '+str(len(Wsizes)))
		Wresults[k+1:l,i], _ = WS_evaluate_chromosomes(reg_module,
				T, V, trail_active_wavenumbers,
				use_stored=True)

	# done moving window
	Wresults=Wresults+(Wresults==0)*np.max(Wresults) # set empty datapoints to max value
	j,i=np.unravel_index(Wresults.argmin(), Wresults.shape)
	bestVal=Wresults[j,i]
	bestSize=Wsizes[i]
	bestStart=j-bestSize//2

	# plot MWresults
	Wresults=np.array(Wresults)
	# make plot
	Wwindowsize,Wwavenumbers = np.meshgrid(Wsizes*abs(dw), wavenumbers)
	unique_keywords=PLSRsave.get_unique_keywords_formatted(common_variables.keyword_lists,keywords)
	PLSRsave.PcolorMW(Wwavenumbers,Wwindowsize,Wresults,fns.add_axis(common_variables.fig,ui['fig_per_row'],ui['max_plots']),unique_keywords[1:],ui)
	if ui['save_check_var']:
		tempCbar=PLSRsave.PcolorMW(Wwavenumbers,Wwindowsize,Wresults,common_variables.tempax,unique_keywords[1:],ui)
		common_variables.tempfig.subplots_adjust(bottom=0.13,left=0.15, right=0.97, top=0.9)
		plotFileName=folder+ui['reg_type']+unique_keywords.replace('.','p')+'_moving_window'
		common_variables.tempfig.savefig(plotFileName+ui['file_extension'])
		tempCbar.remove()
	# set result as keywords, so that they are saved
	bestEnd=bestStart+bestSize
	Wwidth=wavenumbers[bestStart]-wavenumbers[bestEnd-1] #cm-1
	Wcenter=0.5*(wavenumbers[bestStart]+wavenumbers[bestEnd-1]) #cm-1
	keywords['MW width']=str(round(Wwidth,1))+r' cm$^{-1}$'
	keywords['MW center']=str(round(Wcenter,1))+r' cm$^{-1}$'
	# prepare return vector
	active_wavenumers=np.zeros(len(wavenumbers), dtype=bool)
	active_wavenumers[bestStart:bestEnd]=True
	return active_wavenumers

def WS_evaluate_chromosomes(reg_module,T,V,trail_active_wavenumbers,ui=None,use_stored=False,backup_reg_module=None):
	used_mlr=False
	losses=np.zeros(len(trail_active_wavenumbers))
	for i,active_wavenumers in enumerate(trail_active_wavenumbers):
		#print(,i,' of ',len(active_wavenumers))
		#i+=1
		try:
			losses[i]=WS_getRMSEP(reg_module,active_wavenumers,T,V,ui=ui,use_stored=use_stored)
		except:
			used_mlr=True
			losses[i]=WS_getRMSEP(backup_reg_module,active_wavenumers,T,V,ui=ui,use_stored=use_stored)
	return losses, used_mlr


def WS_getRMSEP(reg_module,chromosome,T,V,ui=None,use_stored=False):
	# ui is optional only if use_stored=True
	Ts,Vs=WS_getCrossvalSplits(chromosome,T,V,ui=None,use_stored=use_stored)
	RMSEP=[]
	percent_cor_classified_list=[]
	for curT,curV in zip(Ts,Vs):
		reg_module.fit(curT.X, curT.Y)
		curV.pred = reg_module.predict(curV.X)[:,0]
		if reg_module.type=='regression':
			RMSEP.append(np.sqrt((np.sum((curV.pred-curV.Y)**2)))/len(curV.Y))
		else: #reg_module.type=='classifier'
			percent_cor_classified_list.append(PLSRclassifiers.get_correct_categorized(curV.pred,curV.Y))
	if reg_module.type=='regression':
		return np.sqrt(np.sum(np.array(RMSEP)**2)/len(RMSEP))
	else:
		return 1-np.average(percent_cor_classified_list)

def WS_getCrossvalSplits(chromosome,T,V,ui=None,use_stored=False):
	global stored_XvalTs
	global stored_XvalVs
	if use_stored==True:
		XvalTs = copy.deepcopy(stored_XvalTs)
		XvalVs = copy.deepcopy(stored_XvalVs)
	else:
		XvalTs=[]
		XvalVs=[]
		if ui['WS_loss_type']=='X-validation on training':
			if ui['WS_cross_val_N']==1 and ui['WS_cross_val_max_cases']==-1:
				splitmodule=sklearn.model_selection.LeaveOneOut()
			else:
				splitmodule=sklearn.model_selection.ShuffleSplit(n_splits=ui['WS_cross_val_max_cases'], test_size=ui['WS_cross_val_N'])
			for train,val in splitmodule.split(T.X):
				XvalTs.append(types.SimpleNamespace())
				XvalTs[-1].X=np.array(T.X[train])
				XvalTs[-1].Y=np.array(T.Y[train])
				XvalVs.append(types.SimpleNamespace())
				XvalVs[-1].X=np.array(T.X[val])
				XvalVs[-1].Y=np.array(T.Y[val])
		elif ui['WS_loss_type']=='RMSEC on training':
			XvalTs.append(copy.deepcopy(T))
			XvalVs=XvalTs # pointer to object, no need to copy it
		else:# ui['WS_loss_type']=='RMSEP on validation':
			XvalTs.append(copy.deepcopy(T))
			XvalVs.append(copy.deepcopy(V))
		stored_XvalTs = copy.deepcopy(XvalTs)
		stored_XvalVs = copy.deepcopy(XvalVs)

	for T in XvalTs:
		T.X=T.X[:,chromosome]
	if len(XvalVs[0].X[0])>len(XvalTs[0].X[0]): # this is just a check to see if T==V, in that case we should not act on
		for V in XvalVs:
			V.X=V.X[:,chromosome]
	return XvalTs,XvalVs
