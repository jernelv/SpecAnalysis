import sys
import fns
import numpy as np
import os
def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

def custom_round(f,n):
	if f<10**-2:
		return str(np.format_float_scientific(f,precision=2,exp_digits=1))
	return str(round(f,n))
def get_buttons():
	buttons=[
	{'key': 'RNNtab4name', 'type': 'tabname', 'text': 'Plot Options', 'tab': 4} ,

	{'key': 'verbose', 'type': 'txt:int:range', 'text': 'Text lines', 'default': '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20', 'width': 50, 'tab': 4, 'row': 4} ,
	{'key': 'grid', 'type': 'check', 'text': 'Grid', 'tab': 4, 'row': 4} ,

	{'key': 'fontsize', 'type': 'txt:int', 'text': 'Font size', 'default': '12', 'width': 4, 'tab': 4, 'row': 5} ,
	{'key': 'fig_width', 'type': 'txt:float', 'text': 'Figure Width [inches]', 'default': '6', 'width': 4, 'tab': 4, 'row': 5} ,
	{'key': 'fig_height', 'type': 'txt:float', 'text': 'Figure Height [inches]', 'default': '4', 'width': 4, 'tab': 4, 'row': 5} ,

	{'key': 'spectral_label', 'type': 'label', 'text': 'Spectra plot labels: ', 'tab': 4, 'row': 6} ,
	{'key': 'spec_x_axis_label', 'type': 'txt', 'text': 'x axis', 'default': r'Wavenumber [cm$^{-1}$]', 'width': 30, 'tab': 4, 'row': 6} ,
	{'key': 'spec_y_axis_label', 'type': 'txt', 'text': 'y axis', 'default': r'Absorbance', 'width': 30, 'tab': 4, 'row': 6} ,

	{'key': 'Regression_label', 'type': 'label', 'text': 'Regression plot labels: ', 'tab': 4, 'row': 7} ,
	{'key': 'reg_x_axis_label', 'type': 'txt', 'text': 'x axis', 'default': r'Reference concentration %unit%', 'width': 30, 'tab': 4, 'row': 7} ,
	{'key': 'reg_y_axis_label', 'type': 'txt', 'text': 'y axis', 'default': r'Predicted concentration %unit%', 'width': 30, 'tab': 4, 'row': 7} ,

	{'key': 'Regression_plot_type_label', 'type': 'label', 'text': 'Regression plot axes: ', 'tab': 4, 'row': 8} ,
	{'key': 'reg_plot_type', 'type': 'radio:text', 'texts': ['lin', 'log'], 'tab': 4, 'row': 8} ,


	]
	return buttons
def PlotAbsorbance(ax,fig,active_wavenumers,ui,wavenumbers,training,validation=[],dercase=None):
	ax.cla()
	datapointlists=get_datapointlists(active_wavenumers)
	dpoints=datapointlists[0]
	ax.plot(wavenumbers[dpoints],training[0][dpoints],'-',color=[0,0,1],label='Training')
	for i in range(1,len(training)):
		ax.plot(wavenumbers[dpoints],training[i][dpoints],'-',color=[0,0,1])
	if  len(validation)>0:
		ax.plot(wavenumbers[dpoints],validation[0][dpoints],'-',color=[1,0,0],label='Validation')
		for i in range(1,len(validation)):
			ax.plot(wavenumbers[dpoints],validation[i][dpoints],'-',color=[1,0,0])
	if len(datapointlists)>1:
		for dpoints in datapointlists[1:]:
			for i in range(len(training)):
				ax.plot(wavenumbers[dpoints],training[i][dpoints],'-',color=[0,0,1])
			if ui['is_validation']=='Training and Validation':
				for i in range(len(validation)):
					ax.plot(wavenumbers[dpoints],validation[i][dpoints],'-',color=[1,0,0])
	ax.invert_xaxis()
	ax.set_xlabel(ui['spec_x_axis_label'],fontsize=ui['fontsize'])
	ax.set_ylabel(ui['spec_y_axis_label'],fontsize=ui['fontsize'])
	if ui['grid']:
		ax.grid(color=[0.7,0.7,0.7])
	ax.legend(ncol=1,loc=2,fontsize=ui['fontsize'])
	if not dercase==None:
		sg_config=dercase.sg_config
		curDerivative=dercase.derrivative
		xlimits=ax.get_xlim()
		ylimits=ax.get_ylim()
		ypos=0.95
		dy=-0.05
		for s in dercase.preprocessing_done:
			ax.text(get_pos(xlimits,0.6),get_pos(ylimits,ypos),s)
			ypos+=dy

		'''if dercase.used_sg:
			ax.text(get_pos(xlimits,0.6),get_pos(ylimits,ypos),'SGFilterSize: '+str(sg_config.curSGFiltersize))
			ypos+=dy
			ax.text(get_pos(xlimits,0.6),get_pos(ylimits,ypos),'SGOrder: '+str(sg_config.curSGOrder))
			ypos+=dy
			if curDerivative==1:
				ax.text(get_pos(xlimits,0.6),get_pos(ylimits,ypos),'SG first Derivative')
				ypos+=dy
			elif curDerivative==2:
				ax.text(get_pos(xlimits,0.6),get_pos(ylimits,ypos),'SG second Derivative')
				ypos+=dy
		else:
			if curDerivative==1:
				ax.text(get_pos(xlimits,0.6),get_pos(ylimits,ypos),'First Derivative')
				ypos+=dy
			elif curDerivative==2:
				ax.text(get_pos(xlimits,0.6),get_pos(ylimits,ypos),'Second Derivative')
				ypos+=dy
		if ui['normalize']:
			ax.text(get_pos(xlimits,0.6),get_pos(ylimits,ypos),'Normalized individual spectra')
			ypos+=dy'''

def get_datapointlists(active_wavenumers):
	datapointlists=[]
	not_yet_added_datapoints=[]
	for i, wavenum in enumerate(active_wavenumers):
		if wavenum==False:
			if len(not_yet_added_datapoints)>0:
				datapointlists.append(np.array(not_yet_added_datapoints))
				not_yet_added_datapoints=[]
		else:
			not_yet_added_datapoints.append(i)
	if len(not_yet_added_datapoints)>0:
		datapointlists.append(np.array(not_yet_added_datapoints))
	return datapointlists

def get_or_make_absorbance_ax(run):
	ui=run.ui
	ui['fig_per_row']=int(run.frame.buttons['fig_per_row'].get())
	ui['max_plots']=int(run.frame.buttons['max_plots'].get())
	wavenumbers=run.original_wavenumbers
	common_variables=run.common_variables
	fig=common_variables.fig
	for ax in fig.axes:
		if hasattr(ax,'plot_type') and ax.plot_type=='absorbance':
			return ax
	ax=fns.add_axis(fig,ui['fig_per_row'],ui['max_plots'])
	datapoints=common_variables.datapoints
	PlotAbsorbance(ax,fig,datapoints,ui,wavenumbers,common_variables.original_T.X,common_variables.original_V.X)
	ax.plot_type='absorbance'
	return ax

def make_tempfig(ui,frame):
	eprint('making plot for saving files')
	tempfig = frame.hidden_figure
	tempfig.set_size_inches(ui['fig_width'], ui['fig_height'])
	tempfig.set_dpi(ui['DPI'])
	tempfig.clf()
	tempax = tempfig.add_subplot(1, 1, 1)
	tempfig.subplots_adjust(bottom=0.13,left=0.15, right=0.97, top=0.97)
	return tempfig, tempax

def add_feature_importance_twinx(ax,common_variables,ui,wavenumbers,feature_importance):
	if not hasattr(ax,'myt_twinx'):
		ax.myt_twinx=ax.twinx()
	else:
		ax.myt_twinx.cla()
	ax.myt_twinx.set_ylabel('Feature importance',fontsize=ui['fontsize'])
	add_feature_importance(ax.myt_twinx,common_variables,wavenumbers,feature_importance)
	return ax.myt_twinx

def add_feature_importance(ax,common_variables,xax,feature_importance):
	width=np.abs(xax[1]-xax[0])
	#print(len(xax),len(feature_importance))
	ax.bar(xax,feature_importance,width=width,color=[0,0,0],linewidth=0.5,edgecolor=[0,0,0,1])

def plot_feature_importance(ax,common_variables,ui,xax,feature_importance):
	ax.cla()
	add_feature_importance(ax,common_variables,xax,feature_importance)
	ax.invert_xaxis()
	ax.set_xlabel(ui['spec_x_axis_label'],fontsize=ui['fontsize'])
	ax.set_ylabel('Feature importance',fontsize=ui['fontsize'])
	if ui['grid']:
		ax.grid(color=[0.7,0.7,0.7])

def plot_component(ax,ui,wavenum,yax_label,component,xax_label=True):
	ax.plot(wavenum,component,color=[0,0,0,1])
	if not xax_label==None:
		ax.set_xlabel(ui['spec_x_axis_label'],fontsize=ui['fontsize'])
		ax.invert_xaxis()
	ax.set_ylabel(yax_label,fontsize=ui['fontsize'])
	if ui['grid']:
		ax.grid(color=[0.7,0.7,0.7])

def plot_scatter(ax,fig,X,Y,Z,ui,xax_label,yax_label):
	sc=ax.scatter(X,Y,c=Z,cmap='viridis')
	cbar=fig.colorbar(sc, ax=ax)
	ax.set_ylabel(yax_label)
	ax.set_xlabel(xax_label)
	cbar.set_label('Reference Concentration '+ui['unit'])
	if ui['grid']:
		ax.grid(color=[0.7,0.7,0.7])
	return cbar

def plot_component_weights(ax,ui,linreg_coef):
	ax.bar(np.arange(len(linreg_coef))+1,linreg_coef,color=[0,0,0],linewidth=0.5,edgecolor=[0,0,0,1])
	ax.set_xlabel(r'Component',fontsize=ui['fontsize'])
	ax.set_ylabel('Weight',fontsize=ui['fontsize'])
	if ui['grid']:
		ax.grid(color=[0.7,0.7,0.7])
def plot_component_weights_twinx(ax,ui,wavenum,yax_label,component):
	if not hasattr(ax,',myt_twinx'):
		ax.myt_twinx=ax.twinx()
	else:
		ax.myt_twinx.cla()
	ax.myt_twinx.set_ylabel('Feature importance',fontsize=ui['fontsize'])
	plot_component(ax.myt_twinx,ui,wavenum,yax_label,component,xax_label=None)
	return ax.myt_twinx

def PlotChromosomes(ax,wavenumbers,chromosomes,ui,ylabel='Generation'):
	ax.cla()
	ax.invert_yaxis()
	ax.invert_xaxis()
	dw=(wavenumbers[2]-wavenumbers[1])/2
	for i, chromosome in enumerate(chromosomes):
		PlotChromosome(ax,wavenumbers,chromosome,i)
	ax.set_xlim([min(wavenumbers)-dw,max(wavenumbers)+dw])
	ax.set_xlabel(ui['spec_x_axis_label'],fontsize=ui['fontsize'])
	ax.invert_xaxis()
	ax.set_ylabel(ylabel,fontsize=ui['fontsize'])

def PlotChromosome(ax,wavenumbers,chromosome,i,color=[0,0,0,1]):
	dw=(wavenumbers[2]-wavenumbers[1])/2
	X=wavenumbers[chromosome]
	xx=[]
	for x in X:
		xx.append([x-dw,x+dw])
	# join neighbouring
	new_xx=[]
	j=0
	head=0
	tail=head
	while tail<len(xx)-1:
		if abs(xx[tail][1]-xx[tail+1][0])>abs(dw):
			new_xx.append([xx[head][0],xx[tail][1]])
			head=tail+1
			tail=head
		else:
			tail+=1
	new_xx.append([xx[head][0],xx[tail][1]])
	xx=new_xx
	xx=np.array(xx)
	yy=np.ones(xx.shape)*i
	for x,y in zip(xx,yy):
		ax.plot(x,y,'-',color=color)

def plot_regression(Xval_case,case,ui,ax,keywords,RMSe, coeff_det,frac_cor_lab=-1):
	T=Xval_case.T
	V=Xval_case.V
	sg_config=case.sg_config
	curDerivative=case.derrivative
	ax.cla()
	plot_type=ui['reg_plot_type']
	if plot_type=='lin':
		plot_fun=ax.plot
	else: # log
		plot_fun=ax.loglog
		num_ref_le_0=np.sum(np.array(T.Y)<=0)+np.sum(np.array(V.Y)<=0)
		num_pred_le_0=np.sum(np.array(T.pred)<=0)+np.sum(np.array(V.pred)<=0)
		if num_ref_le_0>0:
			print(str(num_ref_le_0)+' values with reference concentration negative or zero. (Could not be included in log plot)')
		if num_pred_le_0>0:
			print(str(num_pred_le_0)+' values with predicted concentration negative or zero. (Could not be included in log plot)')
	plot_fun(T.Y,T.pred,'o',color=[0,0,1],label='Training data')
	if len(V.Y)>0:
		plot_fun(V.Y,V.pred,'*',color=[1,0,0],label='Validation data')
	xlabel=ui['reg_x_axis_label'].replace('%unit%','['+ui['unit']+']')
	ylabel=ui['reg_y_axis_label'].replace('%unit%','['+ui['unit']+']')
	ax.set_ylabel(ylabel,fontsize=ui['fontsize'])
	ax.set_xlabel(xlabel,fontsize=ui['fontsize'])
	ax.legend(ncol=1,loc=2,fontsize=ui['fontsize'])
	#ax.set_title=(ui['cur_control_string'])
	#print(ui['cur_control_string'])
	#ax.set_title('RMS = '+str(round(RMSe,1)),fontsize=ui['fontsize'])
	#ax.axis('equal')
	#ax.set_aspect('auto')
	xlimits=ax.get_xlim()
	ylimits=ax.get_ylim()
	if xlimits[0]>xlimits[1]: #axis has been reversed, reverse back
		ax.set_xlim(xlimits[1],xlimits[0])
		xlimits=ax.get_xlim()
	if 0: # set x-axis to have same resolution as y-axis, so that a the diagonal forms 45 degrees
		ylimits=ax.get_ylim()
		bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
		#bbox=ax.get_tightbbox(ax.figure.canvas.get_renderer())
		xlen=bbox.x1-bbox.x0
		ylen=bbox.y1-bbox.y0
		ax.set_xlim([ylimits[0],ylimits[0]+(ylimits[1]-ylimits[0])*xlen/ylen])
	dy=0.05
	ypos=0.05
	i=1
	if i in ui['verbose']:
		if frac_cor_lab==-1: # not a classifier
			if ui['coeff_det_type']=='R^2':
				ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'R^2 = '+custom_round(coeff_det,4))
			elif ui['coeff_det_type']=='R':
				ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'R = '+custom_round(coeff_det,4))
			ypos+=dy
	if i in ui['verbose']:
		if ui['SEP_MAE_or_%MAE']=='SEP':
			ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'SEP = '+custom_round(Xval_case.SEP,4))
		if ui['SEP_MAE_or_%MAE']=='MAE':
			ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'MAE = '+custom_round(Xval_case.mean_absolute_error,4))
		if ui['SEP_MAE_or_%MAE']=='%MAE':
			ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'%MAE = '+custom_round(Xval_case.mean_absolute_error_percent,2)+' %')
		ypos+=dy
	i+=1
	if i in ui['verbose']:
		ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),ui['cur_control_string'])
		ypos+=dy
	i+=1
	if i in ui['verbose']:
		ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),ui['reg_type'])
		ypos+=dy
	i+=1
	for s in case.preprocessing_done:
		ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),s)
		ypos+=dy
	'''
	if i in ui['verbose']:
		if curDerivative==1:
			ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'First Derivative')
			ypos+=dy
		elif curDerivative==2:
			ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'Second Derivative')
			ypos+=dy
	i+=1
	if ui['use_SG']:
		if i in ui['verbose']:
			ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'SGFilterSize: '+str(sg_config.curSGFiltersize))
			ypos+=dy
		i+=1
		if i in ui['verbose']:
			ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'SGOrder: '+str(sg_config.curSGOrder))
			ypos+=dy
		i+=1
	if ui['filter'] == 'MA':
			#x.text((xlimits[1]-xlimits[0])*0.6+xlimits[0],(ylimits[1]-ylimits[0])*0.15+ylimits[0],'MA filter, n: '+str(ui['filterN']))
			ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'MA filter, n: '+str(ui['filterN']))
			ypos+=dy
	if ui['filter'] == 'Butterworth':
			#ax.text((xlimits[1]-xlimits[0])*0.6+xlimits[0],(ylimits[1]-ylimits[0])*0.15+ylimits[0],'Butterworth, n: '+str(ui['filterN']))
			ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'Butterworth, n: '+str(ui['filterN']))
			ypos+=dy
	if ui['filter'] == 'Hamming':
			#ax.text((xlimits[1]-xlimits[0])*0.6+xlimits[0],(ylimits[1]-ylimits[0])*0.15+ylimits[0],'Hamming, n: '+str(ui['filterN']))
			ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'Hamming, n: '+str(ui['filterN']))
			ypos+=dy
	if ui['fourier_filter']:
		ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'Fourier cut: '+str(ui['fourier_filter_cut']))
		ypos+=dy
		if not ui['fourier_window']=='None':
			ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'fourier window: '+str(ui['fourier_window']))
			ypos+=dy
			ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'window multiplier: '+custom_round(ui['fourier_window_size_multiplier'],2))
			ypos+=dy
			ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'reverse fourier window: '+str(ui['reverse_fourier_window']))
			ypos+=dy'''
	if i in ui['verbose']:
		if frac_cor_lab==-1:
			if ui['is_validation']=='X-val on training':
				xvalRMSE=np.sqrt(np.sum(np.array(case.XvalRMSEs)**2)/len(case.XvalRMSEs))
				#if ui['RMS_type']=='Combined RMSEP+RMSEC':
				#	ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'X-val RMSEC+RMSEP = '+custom_round(xvalRMSE,1)+' '+ui['unit'])
				if ui['RMS_type']=='RMSEC':
					ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'X-val RMSEC = '+custom_round(xvalRMSE,4)+' '+ui['unit'])
				elif ui['RMS_type']=='RMSEP':
					ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'X-val RMSEP = '+custom_round(xvalRMSE,4)+' '+ui['unit'])
			else:
				#if ui['RMS_type']=='Combined RMSEP+RMSEC':
				#	ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'RMSEC+RMSEP = '+custom_round(RMSe,1)+' '+ui['unit'])
				if ui['RMS_type']=='RMSEC':
					ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'RMSEC = '+custom_round(RMSe,4)+' '+ui['unit'])
				elif ui['RMS_type']=='RMSEP':
					ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'RMSEP = '+custom_round(RMSe,4)+' '+ui['unit'])
		else: #classifier
			if ui['is_validation']=='X-val on training':
				xvalCorrClas=np.average(case.XvalCorrClass)
				#if ui['RMS_type']=='Combined RMSEP+RMSEC':
				#	ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'X-val RMSEC+RMSEP = '+custom_round(xvalRMSE,1)+' '+ui['unit'])
				if ui['RMS_type']=='RMSEC':
					ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'X-val calib correctly classified = '+custom_round(xvalCorrClas*100,1)+' %')
				elif ui['RMS_type']=='RMSEP':
					ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'X-val pred correctly classified = '+custom_round(xvalCorrClas*100,1)+' %')
			else:
				#if ui['RMS_type']=='Combined RMSEP+RMSEC':
				#	ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'RMSEC+RMSEP = '+custom_round(RMSe,1)+' '+ui['unit'])
				if ui['RMS_type']=='RMSEC':
					ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'Calib correctly classified = '+custom_round(frac_cor_lab*100,1)+' %')
				elif ui['RMS_type']=='RMSEP':
					ax.text(get_pos(xlimits,0.6,plot_type),get_pos(ylimits,ypos,plot_type),'Pred correctly classified = '+custom_round(frac_cor_lab*100,1)+' %')

		ypos+=dy
	i+=1
	ypos=0.8
	dy=-0.05
	for keyword in keywords:
		if i in ui['verbose']:
			ax.text(get_pos(xlimits,0.05,plot_type),get_pos(ylimits,ypos,plot_type),keyword+' = '+str(keywords[keyword]))
			ypos+=dy
		i+=1

	if ui['grid']:
		ax.grid(color=[0.7,0.7,0.7])
	return


def add_line_to_logfile(filename,Xval_case,case,ui,keywords,RMSe,coeff_det,frac_cor_lab=-1):
	#T=Xval_case.T
	#V=Xval_case.V
	sg_config=case.sg_config
	curDerivative=case.derrivative
	outstr=''
	if frac_cor_lab==-1:
		if ui['is_validation']=='X-val on training':
			xvalRMSE=np.sqrt(np.sum(np.array(case.XvalRMSEs)**2)/len(case.XvalRMSEs))
			if ui['RMS_type']=='RMSEC':
				outstr+='X-val RMSEC '+ui['unit']+'\t'+custom_round(xvalRMSE,4)+'\t'
			elif ui['RMS_type']=='RMSEP':
				outstr+='X-val RMSEP '+ui['unit']+'\t'+custom_round(xvalRMSE,4)+'\t'
		else:
			if ui['RMS_type']=='RMSEC':
				outstr+='RMSEC '+ui['unit']+'\t'+custom_round(RMSe,4)+'\t'
			elif ui['RMS_type']=='RMSEP':
				outstr+='RMSEP '+ui['unit']+'\t'+custom_round(RMSe,4)+'\t'
	else: #classifier
		if ui['is_validation']=='X-val on training':
			xvalCorrClas=np.average(case.XvalCorrClass)
			if ui['RMS_type']=='RMSEC':
				outstr+='X-val calib correctly classified %\t'+custom_round(xvalCorrClas*100,1)+'\t'
			elif ui['RMS_type']=='RMSEP':
				outstr+='X-val pred correctly classified %\t'+custom_round(xvalCorrClas*100,1)+'\t'
		else:
			if ui['RMS_type']=='RMSEC':
				outstr+='Calib correctly classified %\t'+custom_round(frac_cor_lab*100,1)+'\t'
			elif ui['RMS_type']=='RMSEP':
				outstr+='Pred correctly classified %\t'+custom_round(frac_cor_lab*100,1)+'\t'
	if frac_cor_lab==-1: # not a classifier
		if ui['coeff_det_type']=='R^2':
			outstr+='R^2\t'+custom_round(coeff_det,4)+'\t'
		elif ui['coeff_det_type']=='R':
			outstr+='R\t'+custom_round(coeff_det,4)+'\t'
	outstr+='SEP\t'+custom_round(Xval_case.SEP,4)+'\t'
	outstr+='MAE\t'+custom_round(Xval_case.mean_absolute_error,4)+'\t'
	outstr+='%MAE\t'+custom_round(Xval_case.mean_absolute_error_percent,4)+'\t'
	outstr+=ui['cur_control_string']+'\t'
	outstr+=ui['reg_type']+'\t'
	'''
	if curDerivative==1:
		outstr+='First Derivative'+'\t'
	elif curDerivative==2:
		outstr+='Second Derivative'+'\t'
	if ui['use_SG']:
			outstr+='SGFilterSize: '+str(sg_config.curSGFiltersize)+'\t'
			outstr+='SGOrder: '+str(sg_config.curSGOrder)+'\t'
	if ui['filter'] == 'MA':
			outstr+='MA filter, n: '+str(ui['filterN'])+'\t'
	if ui['filter'] == 'Butterworth':
			outstr+='Butterworth, n: '+str(ui['filterN'])+'\t'
	if ui['filter'] == 'Hamming':
			outstr+='Hamming, n: '+str(ui['filterN'])+'\t'

	if ui['filter'] == 'Fourier':
		outstr+='Fourier cut\t'+str(ui['fourier_filter_cut'])+'\t'
		if not ui['fourier_window']=='None':
			outstr+='fourier window\t'+str(ui['fourier_window'])+'\t'
			outstr+='window multiplier\t'+custom_round(ui['fourier_window_size_multiplier'],2)+'\t'
			outstr+='reverse fourier window\t'+str(ui['reverse_fourier_window'])+'\t'
	'''
	for keyword in keywords:
		outstr+=keyword+'\t'+str(keywords[keyword])+'\t'
	for prepros in case.preprocessing_done:
		outstr+=prepros+'\t'
	outstr+=case.folder

	if os.path.exists(filename):
	    append_write = 'a' # append if already exists
	else:
	    append_write = 'w' # make a new file if not
	f = open(filename,append_write)
	f.write(outstr+ '\n')
	f.close()
	return

def get_pos(limits,x,plot_type='lin'):
	if plot_type=='lin':
		return (limits[1]-limits[0])*x+limits[0]
	else: #log
		return 10**((np.log10(limits[1])-np.log10(limits[0]))*x+np.log10(limits[0]))
def get_unique_keywords_formatted(keyword_lists,keywords):
	filename='_' #+'Best'+'Width'+str(round(Wwidth,1))+'Center'+str(round(Wcenter,1))#+str(components)
	for key in keywords.keys():
		if key in keyword_lists:
			if len(keyword_lists[key])>1:
				nice_keyword=str(keywords[key])
				if '.' in nice_keyword:
					nice_keyword=nice_keyword.split('.')
					nice_keyword[-1]=nice_keyword[-1][:min(len(nice_keyword[-1]),3)]
					nice_keyword='.'.join(nice_keyword)
				filename+=key.replace(' ','_')+'_'+nice_keyword+'_'
		else:
			nice_keyword=str(keywords[key])
			filename+=key.replace(' ','_')+'_'+nice_keyword+'_'

	return filename[:-1]

import numpy as np
import matplotlib.pyplot as plt

def PcolorMW(Wwavenumbers,Wwindowsize,Wresults,curax,title,ui):
	xlimits=curax.get_xlim()
	if xlimits[0]<xlimits[1]: #axis is not reversed, invert
		curax.invert_xaxis()
	curax.cla()
	curax.set_aspect('auto')
	cax=curax.pcolor(Wwavenumbers,Wwindowsize,Wresults) #pcolor Wwavenumbers,Wwindowsize,
	curax.set_xlim([np.max(Wwavenumbers),np.min(Wwavenumbers)])
	curax.set_ylim([np.min(Wwindowsize),np.max(Wwindowsize)])
	curax.set_ylabel('Window Width [cm$^{-1}$]',fontsize=ui['fontsize'])
	curax.set_xlabel('Window Center [cm$^{-1}$]',fontsize=ui['fontsize'])
	curax.set_title(title,fontsize=ui['fontsize'])
	cbar=plt.colorbar(cax, ax=curax)
	#	{'key': 'wavelength_selection_loss_type', 'type': 'radio:text', 'texts': ['X-validation on training', 'RMSEC on training', 'RMSEP on validation'], 'tab': 2, 'row': 8} ,
	if ui['WS_loss_type']=='X-validation on training':
		RMStyp='X-val RMSEP'
	elif ui['WS_loss_type']=='RMSEC on training':
		RMStyp='RMSEC'
	elif ui['WS_loss_type']=='RMSEP on validation':
		RMStyp='RMSEP'
	'''if ui['RMS_type']=='Combined RMSEP+RMSEC':
		RMStyp='Combined RMSEC and RMSEP'
	elif ui['RMS_type']=='RMSEC':
		RMStyp='RMSEC'
	elif ui['RMS_type']=='RMSEP':
		RMStyp='RMSEP' '''
	if not ui['reg_type']=='Classifier':
		if ui['WS_loss_type']=='X-validation on training':
			rmse_string='RMSECV'
		elif ui['WS_loss_type']=='RMSEC on training':
			rmse_string='RMSEC'
		else: # ui['WS_loss_type']=='RMSEP on validation':
			rmse_string='RMSEP'
	else:
		if ui['WS_loss_type']=='X-validation on training':
			rmse_string='CV % wrong'
		elif ui['WS_loss_type']=='RMSEC on training':
			rmse_string='calib % wrong'
		else: # ui['WS_loss_type']=='RMSEP on validation':
			rmse_string='pred % wrong'
	cbar.set_label(rmse_string, labelpad=20,rotation=270,fontsize=ui['fontsize']) #270) labelpad=-40, y=1.05,
	return(cbar)


def SaveLogFile(filename,ui,common_variables):
	eprint('save log')
	with open(filename+'/log','w') as f:
		f.write('trainin files:\n')
		for file in common_variables.trainingfiles:
			f.write(file+'\n')
		if hasattr(common_variables,'validationfiles'):
			f.write('\nvalidation files:\n')
			for file in common_variables.validationfiles:
				f.write(file+'\n')
		f.write('\ncurrent regression method: '+ui['reg_type']+'\n')
		f.write('current regression method keywords:\n')
		for key in common_variables.keyword_lists:
			state=str(common_variables.keyword_lists[key])
			f.write(key+': '+state+'\n')
		f.write('\nbutton states:\n')
		for key in ui:
			state=str(ui[key])
			if not '.!mainframe' in state:
				f.write(key+': '+state+'\n')
		'''
		f.write(ui['cur_control_string']+'\n')

		#ui['reg_type']=ui['RMS_type']=='RMSEP'gressionType'].get())-1
		if ui['reg_type']: f.write('Regression tyepe: PCR\n')
		else: f.write('Regression tyepe: PLSR\n')
		#ui['is_validation']=int(buttons['RegressionDataset'].get())-1
		#ui.ComponentsStart=int(buttons['RegressionComponentsStart'].get())
		f.write('Components Start: '+str(ui['components_start'])+'\n')
		#ui.ComponentsEnd=int(buttons['RegressionComponentsEnd'].get())
		f.write('Components End: '+str(ui['components_end'])+'\n')
		f.write('Max Range '+str(ui['max_range'])+'\n')
		#ui['derivative']=int(buttons['RegressionDerivative'].get())-1
		if ui['derivative']=='Not der':
			f.write('Not derivative\n')
		elif ui['derivative']=='1st der':
			f.write('First Derivative\n')
		elif ui['derivative']=='2nd der':
			f.write('Second Derivative\n')
		elif ui['derivative']=='all':
			f.write('Not Derivative, First Derivative, and Second Derivative\n')
		#ui['do_moving_window']=int(buttons['RegressionMW'].get())
		#ui['moving_window_min']=int(buttons['RegressionMWMin'].get())

		if not ui['do_moving_window']:
			f.write('No Moving Window \n')
		else:
			f.write('Moving Window \n')
			f.write('Moving Window min: '+str(ui['moving_window_min'])+'\n')
			#ui['moving_window_max']=int(buttons['RegressionMWMax'].get())
			f.write('Moving Window max: '+str(ui['moving_window_max'])+'\n')
			#ui['use_SG']=int(buttons['RegressionUseSG'].get())
		if not ui['do_genetic_algorithm']:
			f.write('No Genetic Algorithm \n')
		else:
			f.write('GAnumberOfIndividuals '+str(ui['GA_number_of_individuals'])+'\n')
			f.write('GAcrossoverRate '+str(ui['GA_crossover_rate'])+'\n')
			f.write('GAmutationRate '+str(ui['GA_mutation_rate'])+'\n')
			f.write('GAmaxNumberOfGenerations '+str(ui['GA_max_number_of_generations'])+'\n')
			f.write('GAcrossValN '+str(ui['GA_cross_val_N'])+'\n')
			f.write('GAcrossValMaxCases '+str(ui['GAcross_val_max_cases'])+'\n')
		if not ui['use_SG']:
			f.write('No SG filter\n')
		else:
			f.write('SG filter ON\n')
			#ui.SGmin=int(buttons['RegressionSGwindowMin'].get())
			f.write('SG window min: '+str(ui['SG_window_min'])+'\n')
			#ui.SGmax=int(buttons['RegressionSGwindowMax'].get())
			f.write('SG window max: '+str(ui['SG_window_max'])+'\n')
			#ui.SGorderMin=int(buttons['RegressionSGOrderMin'].get())
			f.write('SG order min: '+str(ui['SG_order_min'])+'\n')
			#ui.SGorderMax=int(buttons['RegressionSGOrderMax'].get())
			f.write('SG order max: '+str(ui['SG_order_max'])+'\n')
		f.write('Files for training\tConcentration\n')
		for i in range(len(common_variables.trainingfiles)):
			f.write(common_variables.trainingfiles[i]+'\t'+str(common_variables.T.Y[i])+'\n')
		f.write('Files for validation\tConcentration\n')
		for i in range(len(common_variables.validationfiles)):
			f.write(common_variables.validationfiles[i]+'\t'+str(common_variables.V.Y[i])+'\n')'''
	return
def writePyChemFile(training,trainingTrueValue,validation,validationTrueValue):
	with open('pyChemInput.txt','w') as f:
		for j in range(len(training)):
			for i in range(len(training[0])):
				if i==len(training[0])-1:
					f.write(str(training[j][i]))
				else:
					f.write(str(training[j][i])+'\t')
			f.write('\n')
		for j in range(len(validation)):
			for i in range(len(validation[0])):
				if i==len(validation[0])-1:
					f.write(str(validation[j][i]))
				else:
					f.write(str(validation[j][i])+'\t')
			if not j ==len(validation)-1:
				f.write('\n')
	with open('pyChemMetadata.csv','w') as f:
		f.write('Label,Concentration,Validation\n')
		f.write('Label,Class,Validation\n')
		f.write('0,1,1')
		for R in trainingTrueValue:
			f.write('\n ,'+str(R[0])+',Train')
		for R in validationTrueValue:
			f.write('\n ,'+str(R[0])+',Validation')

def save_active_wavenumbers(filename,wavenumbers,active_wavenumers):
	'''
	saves the active active_wavenumers and wavenumbers in a format similar to dpt.
	filename: path to save to, including extension
	wavenumbers: list of wavenumbers as float
	active_wavenumers: list of True/False to set if that wavenumber should be turned on or not
	'''
	stage=''
	with open(filename,'w') as f:
		for wavenum, active in zip(wavenumbers,active_wavenumers):
			f.write(stage+str(wavenum)+'\t'+str(active))
			stage='\n'

def plot_fourier(ax,fig,T,V,ui):
	ax.cla()
	xax=np.arange(T.X_fft.shape[1])
	if not ui['plot_fourier_log']:
		ax.plot(xax,T.X_fft_uncut[0],'-',color=[0,0,0,0.2])
		ax.plot(xax,T.X_fft[0],'-',color=[0,0,1],label='Training')
		for i in range(1,len(T.X_fft)):
			ax.plot(xax,T.X_fft_uncut[i],'-',color=[0,0,0,0.2])
			ax.plot(xax,T.X_fft[i],'-',color=[0,0,1])
	else:
		ax.semilogy(xax,abs(T.X_fft_uncut[0]),'-',color=[0,0,0,0.2])
		ax.semilogy(xax,abs(T.X_fft[0]),'-',color=[0,0,1],label='Training')
		for i in range(1,len(T.X_fft)):
			ax.semilogy(xax,abs(T.X_fft_uncut[i]),'-',color=[0,0,0,0.2])
			ax.semilogy(xax,abs(T.X_fft[i]),'-',color=[0,0,1])
	if hasattr(V,'X_fft'):
		if not ui['plot_fourier_log']:
			ax.plot(xax,V.X_fft_uncut[0],'-',color=[0,0,0,0.2])
			ax.plot(xax,V.X_fft[0],'-',color=[0,0,1],label='Validation')
			for i in range(1,len(T.X_fft)):
				ax.plot(xax,V.X_fft_uncut[i],'-',color=[0,0,0,0.2])
				ax.plot(xax,V.X_fft[i],'-',color=[0,0,1])
		else:
			ax.semilogy(xax,abs(V.X_fft_uncut[0]),'-',color=[0,0,0,0.2])
			ax.semilogy(xax,abs(V.X_fft[0]),'-',color=[1,0,0],label='Validation')
			for i in range(1,len(V.X_fft)):
				ax.semilogy(xax,abs(V.X_fft_uncut[i]),'-',color=[0,0,0,0.2])
				ax.semilogy(xax,abs(V.X_fft[i]),'-',color=[1,0,0])

	ax.set_xlabel('Inverse'+ui['spec_x_axis_label'],fontsize=ui['fontsize'])
	ax.set_ylabel('Amplitude',fontsize=ui['fontsize'])
	if ui['grid']:
		ax.grid(color=[0.7,0.7,0.7])
	ax.legend(ncol=1,loc=2,fontsize=ui['fontsize'])
