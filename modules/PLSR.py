from __future__ import print_function
import fns
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal
from scipy import signal
#from sklearn.model_selection import LeavePOut
#from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import ElasticNet
import types
from math import sqrt
import copy
import sys
import importlib
from .libs import PLSRsave
from .libs import PLSRGeneticAlgorithm
from .libs import PLSRNN
from .libs import PLSRRNN
from .libs import PLSRCNN
from .libs import PLSR_file_import
from .libs import PLSRregressionMethods
from .libs import PLSRregressionVisualization
from .libs import PLSRpreprocessing
from .libs import PLSRwavelengthSelection
from .libs import PLSRsequential_feature_selectors
from .libs import PLSRclassifiers

def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)


#### this
'''functions_to_wrap = [[matplotlib.axes.Axes,'pcolormesh'],
                     [matplotlib.figure.Figure,'colorbar'],
                     [matplotlib.figure.Figure,'clf'],
                     [matplotlib.figure.Figure,'set_size_inches'],
                     [matplotlib.figure.Figure,'add_subplot'],
                     [matplotlib.figure.Figure,'subplots'],
                     [matplotlib.figure.Figure,'subplots_adjust'],
                     [matplotlib.axes.Axes,'invert_yaxis'],
                     [matplotlib.axes.Axes,'invert_xaxis'],
                     [matplotlib.axes.Axes,'set_title'],
                     [matplotlib.axes.Axes,'axis'],
                     [matplotlib.axes.Axes,'cla'],
                     [matplotlib.axes.Axes,'plot'],
                     [matplotlib.figure.Figure,'savefig'],
                     [matplotlib.axes.Axes,'set_xlim'],
                     [matplotlib.axes.Axes,'set_position'],
                     [matplotlib.axes.Axes,'bar'],
                     [matplotlib.figure.Figure,'add_axes'],
                     [plt,'figure'],
                     ]

for function in functions_to_wrap:
    if not 'function rimt.<locals>.rimt_this' in str(getattr(function[0], function[1])):
    	setattr(function[0], function[1], fns.rimt(getattr(function[0], function[1])))'''


#from multiprocessing import Pool
#import datetime
#matplotlib.rc('text', usetex=True)
#matplotlib.rc('text.latex', preamble=r'\usepackage{upgreek}')




def crossval(T,V,ui,case):
	if not ui['is_validation']=='X-val on training':
		case.supressplot=0
		return [case]
	else:
		case.Xval_cases=[]
		#XvalTs=[]
		#XvalVs=[]
		#supressplots=[]
		if ui['cross_val_N']==1 and ui['cross_val_max_cases']==-1:
			#ui['cross_val_max_cases']=len(T.Y)
			splitodule=LeaveOneOut()
			print('Using sklearn.LeaveOneOut on '+str(len(T.Y))+' measurements. Maxcases set to '+str(len(T.Y)))
		else:
			if ui['cross_val_max_cases']==-1:
				print('cross_val_max_cases set to -1, cross_val_N not set to 1. Setting cross_val_max_cases to default (20)' )
				ui['cross_val_max_cases']=20
			splitodule=ShuffleSplit(n_splits=ui['cross_val_max_cases'], test_size=ui['cross_val_N'])
		for train,val in splitodule.split(T.X):
			case.Xval_cases.append(types.SimpleNamespace())
			case.Xval_cases[-1].train=train
			case.Xval_cases[-1].val=val
			case.Xval_cases[-1].T=types.SimpleNamespace()
			case.Xval_cases[-1].T.X=np.array(T.X[train])
			case.Xval_cases[-1].T.Y=np.array(T.Y[train])
			case.Xval_cases[-1].V=types.SimpleNamespace()
			case.Xval_cases[-1].V.X=np.array(T.X[val])
			case.Xval_cases[-1].V.Y=np.array(T.Y[val])
			case.Xval_cases[-1].supressplot=1
		case.Xval_cases[-1].supressplot=0
	return case.Xval_cases

def run_reg_module(Xval_case,case,ui,common_variables,active_wavenumers,keywords={}):
	T=Xval_case.T
	V=Xval_case.V
	supressplot=Xval_case.supressplot
	wavenumbers=case.wavenumbers
	folder=case.folder
	try:
		keywords=case.keywords
	except:
		keywords={}
		print('let the developers know if you see this error')
	# Set what datapoints to include, the parameter 'wavenum' is in units cm^-1
	#datapointlists=ui.datapointlists

	# common_variables.tempax and common_variables.tempfig are for the figure that is saved, common_variables.ax and common_variables.fig are for the figure that is displayed
	# need to have this for the colorbar
	if ui['save_check_var']:
		common_variables.tempax.fig=common_variables.tempfig
	#plot best result
	# or only result if not MW
	reg_module=PLSRregressionMethods.getRegModule(ui['reg_type'],keywords,ui['scaling'],ui['mean_centering'])
	#reg_module.active_wavenumers=active_wavenumers
	# get RMSe
	for E in [T,V]:
		if len(E.Y)>0:
			E.Xsmol=E.X[:,active_wavenumers]
	reg_module.fit(T.Xsmol, T.Y)
	for E in [T,V]:
		if len(E.Y)>0:
			E.pred = reg_module.predict(E.Xsmol)[:,0]
		else:
			E.pred = []
	Xval_case.RMSECP=np.sqrt((np.sum((T.pred-T.Y)**2)+np.sum((V.pred-V.Y)**2))/(len(T.Y)+len(V.Y)))
	Xval_case.RMSEC=np.sqrt((np.sum((T.pred-T.Y)**2))/(len(T.Y)))
	if len(V.Y)>0:
		Xval_case.RMSEP=np.sqrt((np.sum((V.pred-V.Y)**2))/(len(V.Y)))
	'''if ui['RMS_type']=='Combined RMSEP+RMSEC' and len(V.Y)>0:
		RMSe=Xval_case.RMSECP
		Y_for_r2=np.concatenate((T.Y,V.Y))
		pred_for_r2=np.concatenate((T.pred,V.pred))
	el'''
	if ui['RMS_type']=='RMSEP':
		RMSe=Xval_case.RMSEP
		Y_for_r2=V.Y
		pred_for_r2=V.pred
	else:
		RMSe=Xval_case.RMSEC
		Y_for_r2=T.Y
		pred_for_r2=T.pred

	case.XvalRMSEs.append(RMSe)
	#calculating coefficient of determination
	if not hasattr(case,'X_val_pred'):
		case.X_val_pred=[pred_for_r2]
		case.X_val_Y=[Y_for_r2]
	else:
		case.X_val_pred.append(pred_for_r2)
		case.X_val_Y.append(Y_for_r2)
	if not supressplot: # if plotting this, calculate R^2 for all xval cases
		X_pred=np.array(case.X_val_pred).reshape(-1)
		X_Y=np.array(case.X_val_Y).reshape(-1)
		y_mean = np.sum(X_Y)*(1/len(X_Y))
		Xval_case.R_squared = 1 - ((np.sum((X_Y - X_pred)**2))/(np.sum((X_Y - y_mean)**2)))
		avg=np.average(X_pred-X_Y)
		n=len(X_pred)
		Xval_case.SEP=np.sqrt(np.sum( ( X_pred-X_Y-avg   )**2 )/(n-1))
	else:
		Xval_case.R_squared=0
		Xval_case.SEP=0
	try:
		Xval_case.R_not_squared=sqrt(Xval_case.R_squared)
	except:
		Xval_case.R_not_squared=0
	if ui['coeff_det_type']=='R^2':
		coeff_det = Xval_case.R_squared
	elif ui['coeff_det_type']=='R':
		coeff_det = Xval_case.R_not_squared
	if reg_module.type=='classifier':#'classifier_type' in keywords:
		frac_cor_lab=PLSRclassifiers.get_correct_categorized(case.X_val_Y[-1],case.X_val_pred[-1])
		case.XvalCorrClass.append(frac_cor_lab)
	else:
		frac_cor_lab=-1
	#plot
	if not supressplot:
		PLSRsave.plot_regression(Xval_case,case,ui,fns.add_axis(common_variables.fig,ui['fig_per_row'],ui['max_plots']),keywords,RMSe, coeff_det,frac_cor_lab=frac_cor_lab)
		if ui['save_check_var']:
			if not ui['do_not_save_plots']:
				PLSRsave.plot_regression(Xval_case,case,ui,common_variables.tempax,keywords,RMSe, coeff_det,frac_cor_lab=frac_cor_lab)
				common_variables.tempfig.subplots_adjust(bottom=0.13,left=0.15, right=0.97, top=0.95)
				#common_variables.tempfig.savefig(folder+'Best'+'Comp'+str(components)+'Width'+str(round(Wwidth,1))+'Center'+str(round(Wcenter,1))+'.pdf')
				#common_variables.tempfig.savefig(folder+'Best'+'Comp'+str(components)+'Width'+str(round(Wwidth,1))+'Center'+str(round(Wcenter,1))+'.svg')
				plotFileName=case.folder+ui['reg_type']+PLSRsave.get_unique_keywords_formatted(common_variables.keyword_lists,case.keywords).replace('.','p')
				common_variables.tempfig.savefig(plotFileName+ui['file_extension'])
			PLSRsave.add_line_to_logfile(case.folder+'results_table',Xval_case,case,ui,keywords,RMSe,coeff_det,frac_cor_lab=frac_cor_lab)
		#draw(common_variables)
	return reg_module, RMSe


class moduleClass():
	filetypes=['DPT','dpt','list','txt','laser']
	def __init__(self, fig, locations, frame, ui):
		#reload modules
		if frame.module_reload_var.get():
			if 'modules.libs.PLSRsave' in sys.modules: #reload each time it is run
				importlib.reload(sys.modules['modules.libs.PLSRsave'])
			if 'modules.libs.PLSRGeneticAlgorithm' in sys.modules: #reload each time it is run
				importlib.reload(sys.modules['modules.libs.PLSRGeneticAlgorithm'])
			if 'modules.libs.PLSRsequential_feature_selectors' in sys.modules: #reload each time it is run
				importlib.reload(sys.modules['modules.libs.PLSRsequential_feature_selectors'])
			if 'modules.libs.PLSRNN' in sys.modules: #reload each time it is run
				importlib.reload(sys.modules['modules.libs.PLSRNN'])
			if 'modules.libs.PLSRRNN' in sys.modules: #reload each time it is run
				importlib.reload(sys.modules['modules.libs.PLSRRNN'])
			if 'modules.libs.PLSRCNN' in sys.modules: #reload each time it is run
				importlib.reload(sys.modules['modules.libs.PLSRCNN'])
			if 'modules.libs.PLSR_file_import' in sys.modules: #reload each time it is run
				importlib.reload(sys.modules['modules.libs.PLSR_file_import'])
			if 'modules.libs.PLSRregressionMethods' in sys.modules: #reload each time it is run
				importlib.reload(sys.modules['modules.libs.PLSRregressionMethods'])
			if 'modules.libs.PLSRclassifiers' in sys.modules: #reload each time it is run
				importlib.reload(sys.modules['modules.libs.PLSRclassifiers'])
			if 'modules.libs.PLSRregressionVisualization' in sys.modules: #reload each time it is run
				importlib.reload(sys.modules['modules.libs.PLSRregressionVisualization'])
			if 'modules.libs.PLSRpreprocessing' in sys.modules: #reload each time it is run
				importlib.reload(sys.modules['modules.libs.PLSRpreprocessing'])
			if 'modules.libs.PLSRwavelengthSelection' in sys.modules: #reload each time it is run
				importlib.reload(sys.modules['modules.libs.PLSRwavelengthSelection'])


		#code for checking for memory leaks
		global run #global keyword used to connect button clicks to class object
		run=self
		self.fig=fig
		self.locations=locations
		self.frame=frame
		self.ui=ui

	def run(self):
		fig=self.fig
		locations=self.locations
		frame=self.frame
		ui=self.ui
		eprint('running')
		self.fig=fig
		fig.clf()
		self.frame=frame
		# get variables from buttons
		common_variables=types.SimpleNamespace()
		common_variables.draw=self.draw
		self.common_variables=common_variables
		common_variables.keyword_lists={}

		PLSRregressionMethods.get_relevant_keywords(common_variables,ui)

		ui['multiprocessing']=1-(ui['no_multiprocessing'])

		save_check_var=frame.save_check_var.get()
		ui['save_check_var']=save_check_var
		filename=frame.name_field_string.get()
		self.filename=filename
		#prepare figures for display (set correct number of axes, each pointing to the next axis)
		######################### if crossval and moving window -> stop ###########
		if ui['is_validation']=='X-val on training' and ui['regression_wavelength_selection']=='Moving window':
			print("Use of x-validation with moving window is not supported")
			return
		######################### if RMSEP and no validation -> stop ##############
		if ui['is_validation']=='Training' and ui['RMS_type']=='RMSEP':
			print("Unable to calculate RMSEP with only training set")
			return
		#################### if RMSEP and RMSEC and no validation -> only RMSEP ###
		if ui['is_validation']=='Training':
			ui['RMS_type']='RMSEC'
			if ui['RMS_type']=='Default':
				ui['RMS_type']='RMSEC'
		else:
			if ui['RMS_type']=='Default':
				ui['RMS_type']='RMSEP'

		common_variables.frame=frame
		common_variables.fig=fig
		################################################################################################
		######################### Load data as training or validation ##################################
		################################################################################################
		T=types.SimpleNamespace()
		V=types.SimpleNamespace()
		if len(frame.training_files)==0:
				print('training set required')
				return
		#load training set
		T.X, T.Y, common_variables.trainingfiles, self.wavenumbers, self.regressionCurControlTypes=PLSR_file_import.get_files(frame.training_files,ui['max_range'])
		self.original_wavenumbers=self.wavenumbers

		for i, contrltytpe in enumerate(self.regressionCurControlTypes):
			frame.button_handles['cur_col'][i]["text"]=contrltytpe

		if ui['is_validation']=='Training' or ui['is_validation']=='X-val on training':# if training or crossval -> deselect validation
			frame.nav.deselect()
			#frame.nav.clear_color('color3')
			#frame.validation_files=frame.nav.get_paths_of_selected_items()
			V.X=np.array([]) # set empty validation set
			V.Y=np.array([])

		elif ui['is_validation']=='Training and Validation':
			if len(frame.validation_files)==0:
				print('training and validation set, but no validation set in in put')
				return
			#load validation set
			V.X, V.Y, common_variables.validationfiles, _, _2=PLSR_file_import.get_files(frame.validation_files,ui['max_range'])

		common_variables.original_T=copy.deepcopy(T)
		common_variables.original_V=copy.deepcopy(V)

		################################################################################################
		################################## load reference spectra #######################################
		################################################################################################
		if ui['reference_spectra']=='':
			self.reference_spectra=None
		else:
			try:
				temp, _1, _2, _3, _4=PLSR_file_import.get_files([ui['reference_spectra']],np.inf)
				if len(temp)>0:
					print('first reference spectra in list selected for reference spectra selected as reference spectra')
				self.reference_spectra=np.array(temp[0])
			except Exception as e:
				self.reference_spectra=None
				print(e)
				print('error importing referece spectra -> ignoring')
		if ui['background_spectra']=='':
			self.background_spectra=None
		else:
			try:
				temp, _1, _2, _3, _4=PLSR_file_import.get_files([ui['background_spectra']],np.inf)
				if len(temp)>0:
					print('first background spectra in list selected for reference spectra selected as reference spectra')
				self.background_spectra=np.array(temp[0])
			except Exception as e:
				self.background_spectra=None
				print(e)
				print('error importing referece spectra -> ignoring')

		################################################################################################
		################# set up folder, save log and temporary figure for saving ######################
		################################################################################################

		if save_check_var:
			if not os.path.exists(filename):
				os.makedirs(filename)
			PLSRsave.SaveLogFile(filename,ui,common_variables)
			common_variables.tempfig,common_variables.tempax=PLSRsave.make_tempfig(ui,frame)
		################################################################################################
		############################## calculate window ranges #########################################
		################################################################################################
		common_variables.datapoints=np.arange(len(self.wavenumbers))
		#common_variables.datapointlists=[common_variables.datapoints]# declare this for get_or_make_absorbance_ax
		#common_variables.datapoints, common_variables.datapointlists=PLSRpreprocessing.GetDatapoints(self.wavenumbers, ui)
		################################################################################################
		################################### save unprocessed spectra ###################################
		################################################################################################
		if ui['plot_spectra_before_preprocessing']:
			eprint('plot abs')
			if ui['save_check_var']:
				PLSRsave.PlotAbsorbance(common_variables.tempax,common_variables.tempfig,common_variables.datapoints,ui,self.wavenumbers,T.X,V.X)
				plotFileName=filename+'/SpectraPrePreprocessing'
				common_variables.tempfig.savefig(plotFileName.replace('.','p')+ui['file_extension'])
				common_variables.tempax.cla()
			ax=PLSRsave.get_or_make_absorbance_ax(self)
			self.draw()

		################################################################################################
		################################### make pychem input file #####################################
		################################################################################################
		if int(ui['make_pyChem_input_file']):
			if ui['is_validation']=='Training and Validation':
				PLSRsave.writePyChemFile(T.X,T.Y,validation,validationtruevalues)
			else:
				PLSRsave.writePyChemFile(T.X,T.Y,[],[])
		################################################################################################
		################## set current control and remove data higher than maxrange ####################
		################################################################################################
		datasets=[T]
		if ui['is_validation']=='Training and Validation':
			datasets.append(V)
		for E in datasets:
			keepsamples=[]
			for i,_ in enumerate(E.Y):
				if not E.Y[i,ui['cur_col']] > ui['max_range']:
					keepsamples.append(i)
			E.X=E.X[keepsamples,:]
			E.Y=E.Y[keepsamples,ui['cur_col']]
		ui['cur_control_string']=self.regressionCurControlTypes[ui['cur_col']]

		PLSRpreprocessing.do_preprocessing(self,T,V)
		if ui['plot_fourier']:
			if hasattr(T,'X_fft'):
				ax=fns.add_axis(fig,ui['fig_per_row'],ui['max_plots'])
				PLSRsave.plot_fourier(ax,fig,T,V,ui)

		self.complete_cases=[]
		for _ in [1]: # is a loop so that you can use 'break'
			for i,dercase in enumerate(self.preprocessed_cases):
				#need to set data range in case of derrivative, rerunn in all cases anyways
				datapoints=PLSRpreprocessing.GetDatapoints(dercase.wavenumbers, ui)
				#common_variables.datapoints=datapoints
				#common_variables.datapointlists=datapointlists
				if ui['plot_spectra_after_preprocessing']:
					ax=fns.add_axis(fig,ui['fig_per_row'],ui['max_plots'])
					PLSRsave.PlotAbsorbance(ax,fig,datapoints,ui,dercase.wavenumbers,dercase.T.X,dercase.V.X,dercase=dercase)
					self.draw()
					if ui['save_check_var']:
						PLSRsave.PlotAbsorbance(common_variables.tempax,common_variables.tempfig,datapoints,ui,dercase.wavenumbers,dercase.T.X,dercase.V.X,dercase=dercase)
						plotFileName=dercase.folder+'/SpectraPostPreprocessing'
						common_variables.tempfig.savefig(plotFileName.replace('.','p')+ui['file_extension'])
						common_variables.tempax.cla()
				for E in [dercase.T,dercase.V]:
					if len(E.Y)>0:
						E.X=E.X[:,datapoints]
				dercase.wavenumbers=dercase.wavenumbers[datapoints]

				#create complete cases for all pemutations of keyword values in keyword_lists
				for keyword_case in PLSRregressionMethods.generate_keyword_cases(common_variables.keyword_lists):
					self.complete_cases.append(types.SimpleNamespace())
					self.complete_cases[-1].wavenumbers=dercase.wavenumbers
					self.complete_cases[-1].folder=dercase.folder
					self.complete_cases[-1].sg_config=dercase.sg_config
					self.complete_cases[-1].derrivative=dercase.derrivative
					self.complete_cases[-1].T=dercase.T
					self.complete_cases[-1].V=dercase.V
					self.complete_cases[-1].keywords=keyword_case
			if ui['reg_type']=='None':
				break
			for case in self.complete_cases:
				case.XvalRMSEs=[]
				case.XvalCorrClass=[]
				common_variables.keywords=case.keywords
				#GeneticAlgorithm(ui,T,V,datapoints,components)
				if ui['regression_wavelength_selection']=='No wavelength selection':
					active_wavenumers = np.ones(len(case.wavenumbers), dtype=bool)
				else:
					# report to user regarding split module
					if self.ui['WS_loss_type']=='X-validation on training':
						if self.ui['WS_cross_val_N']==1 and self.ui['WS_cross_val_max_cases']==-1:
							print('Using sklearn.LeaveOneOut on '+str(len(case.T.Y))+' measurements. Maxcases set to '+str(len(case.T.Y)))
						else:
							if self.ui['WS_cross_val_max_cases']==-1:
								print('WS_cross_val_max_cases set to -1, GA_cross_val_N not set to 1. Setting GAcross_val_max_cases to default (20)' )
								self.ui['WS_cross_val_max_cases']=20
					if ui['regression_wavelength_selection']=='Genetic Algorithm':
						GAobject = PLSRGeneticAlgorithm.GeneticAlgorithm(common_variables,ui,case)
						active_wavenumers = GAobject.run(fns.add_axis(common_variables.fig,ui['fig_per_row'],ui['max_plots']),case.wavenumbers,case.folder,self.draw)
					elif ui['regression_wavelength_selection']=='Moving Window':
						active_wavenumers = PLSRwavelengthSelection.MW(case,ui,common_variables)
					elif ui['regression_wavelength_selection']=='Sequential Feature Selector':
						FSobject = PLSRsequential_feature_selectors.sequentialFeatureSelector(common_variables,ui,case,self.draw)
						active_wavenumers = FSobject.run()



				Xval_cases=crossval(case.T,case.V,ui,case) # returns [T],[V] if not crossva, otherwise makes cases from validation dataset
				for Xval_case in Xval_cases:
					#	ui.datapoints=runGeneticAlgorithm(dercase[0],dercase[1],dercase[2],dercase[3],dercase[4],dercase[5],dercase[6],dercase[7])
					#def MW(T,V,wavenumbers, folder,ui,sg_config,curDerivative,supressplot):
					if  ui['save_check_var']:
						active_wavenumbers_file=case.folder+ui['reg_type']+PLSRsave.get_unique_keywords_formatted(common_variables.keyword_lists,case.keywords).replace('.','p')+'active_wavenumers.dpb'
						PLSRsave.save_active_wavenumbers(active_wavenumbers_file,case.wavenumbers,active_wavenumers)
					case.active_wavenumers=active_wavenumers
					self.draw()
					self.last_reg_module, RMSe = run_reg_module(Xval_case,case,ui,common_variables,active_wavenumers,keywords={})
					self.draw()
					self.last_complete_case = case
					self.last_Xval_case = Xval_case
					if Xval_case.supressplot==0:
						if ui['is_validation']=='X-val on training':
							#if ui['RMS_type']=='Combined RMSEP+RMSEC':
							#	print('RMSEC+RMSEP = '+PLSRsave.custom_round(case.xvalRMSE,3)+' '+ui['unit'])
							if not 'classifier_type' in case.keywords:
								case.xvalRMSE=np.sqrt(np.sum(np.array(case.XvalRMSEs)**2)/len(case.XvalRMSEs))
								if ui['RMS_type']=='RMSEC':
									print('RMSEC = '+PLSRsave.custom_round(case.xvalRMSE,3)+' '+ui['unit'])
								elif ui['RMS_type']=='RMSEP':
									print('RMSEP = '+PLSRsave.custom_round(case.xvalRMSE,3)+' '+ui['unit'])
							else:
								print(case.XvalCorrClass)
								case.xvalCorrClas=np.average(case.XvalCorrClass)
								print(case.xvalCorrClas)
								if ui['RMS_type']=='RMSEC':
									print('x-val corr classifed training = '+str(round(case.xvalCorrClas*100,3))+' %')
								elif ui['RMS_type']=='RMSEP':
									print('x-val corr classifed prediction = '+str(round(case.xvalCorrClas*100,3))+' %')

						case.XvalRMSEs=[]
				eprint('done')
		#plt.close(common_variables.tempfig)
		#del common_variables.tempfig
		if save_check_var:
			# save plot in window
			fig.savefig(filename+'/'+'_'.join(filename.split('/')[1:])+ui['file_extension'])
		print('Done')
		return

	def callbackClick(self,frame,event):
		ax=event.inaxes
		if hasattr(ax,'plot_type'):
			if ax.plot_type=='NN node map':
				PLSRregressionVisualization.plot_node_activation_vector(event)
				return
		else:
			print("clicked at", event.xdata, event.ydata)

	def reorder_plots(self,event):
		ui=self.ui
		ui['fig_per_row']=int(self.frame.buttons['fig_per_row'].get())
		ui['max_plots']=int(self.frame.buttons['max_plots'].get())
		fns.move_all_plots(self.fig,ui['fig_per_row'],ui['max_plots'])
		self.draw()

	@fns.rimt
	def draw(self):
	    self.fig.canvas.draw()
	    self.frame.update()

	def addButtons():
		buttons=[
		{'key': 'RNNtab3name', 'type': 'tabname', 'text': 'Import Options', 'tab': 3} ,

		# dataset configuration
		{'key': 'RegressionL0', 'type': 'label', 'text': 'Data import options: ', 'tab': 3, 'row': 0} ,
		{'key': 'is_validation', 'type': 'radio:text', 'texts': ['Training', 'Training and Validation', 'X-val on training'], 'tab': 3, 'row': 0} ,
		{'key': 'cross_val_N', 'type': 'txt:int', 'text': 'Number of validation samples for cross validation', 'default': '10', 'width': 4, 'tab': 3, 'row': 1} ,
		{'key': 'cross_val_max_cases', 'type': 'txt:int', 'text': 'Iterations', 'default': '-1', 'width': 4, 'tab': 3, 'row': 1} ,
		{'key': 'RegressionL0a', 'type': 'label', 'text': 'Column of data to use: ', 'tab': 3, 'row': 2} ,
		{'key': 'cur_col', 'type': 'radio', 'texts': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], 'tab': 3, 'row': 2} ,
		{'key': 'max_range', 'type': 'txt:float', 'text': 'Maximum concentration for training set', 'default': '10000', 'width': 6, 'tab': 3, 'row': 3} ,
		{'key': 'unit', 'type': 'txt', 'text': 'Concentration unit', 'default': 'mg/dl', 'width': 6, 'tab': 3, 'row': 4} ,

		# config for creating figure and saving
		{'key': 'file_extension', 'type': 'radio:text', 'texts': [ '.svg', '.png', '.pdf'], 'tab': 4, 'row': 1} ,
		{'key': 'reorder_plots', 'type': 'click', 'text': 'Reorder plots', 'bind': reorder_plots, 'tab': 4, 'row': 1} ,
		{'key': 'fig_per_row', 'type': 'txt:int', 'text': 'Figures per row', 'default': '2', 'width': 4, 'tab': 4, 'row': 1} ,
		{'key': 'DPI', 'type': 'txt:int', 'text': 'dpi', 'default': '80', 'width': 4, 'tab': 4, 'row': 1} ,

		# graphical user interface options
		{'key': 'max_plots', 'type': 'txt:int', 'text': 'Max number of plots', 'default': '-1', 'width': 3, 'tab': 4, 'row': 2} ,

		# save options
		{'key': 'make_pyChem_input_file', 'type': 'check', 'text': 'Make pyChem file', 'tab': 4, 'row': 9} ,
		{'key': 'do_not_save_plots', 'type': 'check', 'text': 'do not save plots', 'tab': 4, 'row': 8} ,


		# debugging options
		{'key': 'RNNtab5name', 'type': 'tabname', 'text': 'Other', 'tab': 5} ,
		{'key': 'no_multiprocessing', 'type': 'radio', 'texts': ['use multiprocessing', 'do not use multiprocessing'], 'tab': 5, 'row': 0},

		# result
		{'key': 'RMS_type', 'type': 'radio:text', 'texts': ['Default', 'RMSEC', 'RMSEP'], 'tab': 3, 'row': 9} ,
		{'key': 'coeff_det_type', 'type': 'radio:text', 'texts': ['R^2', 'R'], 'tab': 3, 'row': 9} ,

		# declare input
		{'key': 'set_training', 'type': 'click', 'text': 'Set Training', 'bind': set_training,'color':'color1', 'tab': 10, 'row': 0} ,
		{'key': 'set_validation', 'type': 'click', 'text': 'Set Validation', 'bind': set_validation,'color':'color3', 'tab': 10, 'row': 0} ,
		]
		buttons+=PLSRregressionMethods.get_buttons()
		buttons+=PLSRclassifiers.get_buttons()
		buttons+=PLSRsave.get_buttons()
		buttons+=PLSRwavelengthSelection.get_buttons()
		buttons+=PLSRpreprocessing.get_buttons()
		return buttons

def set_training(event):
    """Sets the training data set(s) in the GUI."""
    frame=event.widget.master.master.master
    frame.nav.clear_color('color1')
    frame.nav.color_selected('color1')
    frame.training_files=frame.nav.get_paths_of_selected_items()
    frame.nav.deselect()
    return

def set_validation(event):
    """Sets the validation data set(s) in the GUI."""
    frame=event.widget.master.master.master
    frame.nav.clear_color('color3')
    frame.nav.color_selected('color3')
    frame.validation_files=frame.nav.get_paths_of_selected_items()
    frame.nav.deselect()
    return

def reorder_plots(event):
	global run
	run.reorder_plots(run,event)
	return
