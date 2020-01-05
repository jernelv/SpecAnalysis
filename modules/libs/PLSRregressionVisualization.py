import scipy
import numpy as np
import fns
from . import PLSRsave
from . import PLSRNN
from . import PLSRCNN
from modules import PLSR

def plot_components_PLSR(event):
	run=PLSR.run
	if not run.ui['reg_type'] == 'PLSR':
		return
	"""Function for making plots of the latent variables from PLSR. """
	ui=run.ui
	ui['fig_per_row']=int(run.frame.buttons['fig_per_row'].get())
	ui['max_plots']=int(run.frame.buttons['max_plots'].get())
	common_variables=run.common_variables
	reg_module=run.last_reg_module
	latent_variables=np.swapaxes(reg_module.x_weights_,0,1)
	#print("this is working")last_reg_module
	#print(latent_variables.shape)
	ui=run.ui
	wavenum=run.last_complete_case.wavenumbers
	if ui['save_check_var']:
		tempax=common_variables.tempax
		tempfig=common_variables.tempfig
	for i, latent_variable in enumerate(latent_variables):
		ax=fns.add_axis(common_variables.fig,ui['fig_per_row'],ui['max_plots'])
		yax_label='Latent variable '+str(i+1)
		PLSRsave.plot_component(ax,ui,wavenum,yax_label,latent_variable)
		run.draw()
		if ui['save_check_var']:
			tempax.cla()
			PLSRsave.plot_component(tempax,ui,wavenum,yax_label,latent_variable)
			plotFileName=run.filename+'/PLSR latent variable '+str(i+1)
			tempfig.savefig(plotFileName.replace('.','p')+ui['file_extension'])
	return

def plot_components_PCR(event):
	run=PLSR.run
	if not run.ui['reg_type'] == 'PCR':
		return
	"""Function for making plots of the principal components from PCR."""
	ui=run.ui
	ui['fig_per_row']=int(run.frame.buttons['fig_per_row'].get())
	ui['max_plots']=int(run.frame.buttons['max_plots'].get())
	common_variables=run.common_variables
	reg_module=run.last_reg_module
	components=reg_module.pca.components_[:reg_module.components]
	#print(components)
	ui=run.ui
	wavenum=run.last_complete_case.wavenumbers
	if ui['save_check_var']:
		tempax=common_variables.tempax
		tempfig=common_variables.tempfig
	for i, component in enumerate(components):
		ax=fns.add_axis(common_variables.fig,ui['fig_per_row'],ui['max_plots'])
		yax_label='Component '+str(i+1)
		PLSRsave.plot_component(ax,ui,wavenum,yax_label,component)
		run.draw()
		if ui['save_check_var']:
			tempax.cla()
			PLSRsave.plot_component(tempax,ui,wavenum,yax_label,component)
			plotFileName=run.filename+'/PCR component '+str(i+1)
			tempfig.savefig(plotFileName.replace('.','p')+ui['file_extension'])
	linreg_coef=reg_module.linreg.coef_
	linreg_coef=linreg_coef/sum(linreg_coef)
	ax=fns.add_axis(common_variables.fig,ui['fig_per_row'],ui['max_plots'])
	PLSRsave.plot_component_weights(ax,ui,linreg_coef)
	if ui['save_check_var']:
		tempax.cla()
		PLSRsave.plot_component_weights(tempax,ui,linreg_coef)
		plotFileName=run.filename+'/PCR Weights'
		tempfig.savefig(plotFileName.replace('.','p')+ui['file_extension'])
	run.draw()

	ax=fns.add_axis(common_variables.fig,ui['fig_per_row'],ui['max_plots'])
	product=np.dot(np.transpose(components),linreg_coef)
	yax_label=r'Comps$\cdot$weights'
	PLSRsave.plot_component(ax,ui,wavenum,yax_label,product)
	if ui['save_check_var']:
		tempax.cla()
		PLSRsave.plot_component(tempax,ui,wavenum,yax_label,product)
		plotFileName=run.filename+'/PCR components times weights'
		tempfig.savefig(plotFileName.replace('.','p')+ui['file_extension'])
	run.draw()

	ax=PLSRsave.get_or_make_absorbance_ax(run)
	PLSRsave.plot_component_weights_twinx(ax,ui,wavenum,yax_label,product)
	if ui['save_check_var']:
		tempax=common_variables.tempax
		tempfig=common_variables.tempfig
		common_variables.tempfig.subplots_adjust(bottom=0.13,left=0.15, right=0.85, top=0.97)
		PLSRsave.PlotAbsorbance(tempax,tempfig,run.last_complete_case.active_wavenumers,ui,run.last_complete_case.wavenumbers,common_variables.original_T.X,common_variables.original_V.X)
		twinx=PLSRsave.plot_component_weights_twinx(tempax,ui,wavenum,yax_label,product)
		plotFileName=run.filename+'/transmission and PCR components times weights '
		tempfig.savefig(plotFileName.replace('.','p')+ui['file_extension'])
		tempax.cla()
		twinx.remove()
		common_variables.tempfig.subplots_adjust(bottom=0.13,left=0.15, right=0.97, top=0.97)
	run.draw()
	return
def plot_PCR_scatter(event):
	run=PLSR.run
	if not run.ui['reg_type'] == 'PCR':
		return
	ui=run.ui
	ui['fig_per_row']=int(run.frame.buttons['fig_per_row'].get())
	ui['max_plots']=int(run.frame.buttons['max_plots'].get())
	ui['PCR_scatter_X']=int(run.frame.buttons['PCR_scatter_X'].get())
	ui['PCR_scatter_Y']=int(run.frame.buttons['PCR_scatter_Y'].get())
	common_variables=run.common_variables
	reg_module = run.last_reg_module
	case = run.last_Xval_case
	if len(case.V.X)>0:
		X_reduced=reg_module.get_X_reduced(case.V.X)
		Z=case.V.Y
	else:
		X_reduced=reg_module.get_X_reduced(case.T.X)
		Z=case.T.Y
	x_com=ui['PCR_scatter_X']
	y_com=ui['PCR_scatter_Y']
	X=X_reduced[:,x_com-1]
	Y=X_reduced[:,y_com-1]
	ax=fns.add_axis(common_variables.fig,ui['fig_per_row'],ui['max_plots'])
	PLSRsave.plot_scatter(ax,run.fig,X,Y,Z,ui,'Component '+str(x_com),'Component '+str(y_com))
	run.draw()
	if ui['save_check_var']:
		tempax=common_variables.tempax
		tempfig=common_variables.tempfig
		tempax.cla()
		common_variables.tempfig.subplots_adjust(bottom=0.13,left=0.15, right=0.85, top=0.97)
		temp_cbar=PLSRsave.plot_scatter(tempax,tempfig,X,Y,Z,ui,'Component '+str(x_com),'Component '+str(y_com))
		plotFileName=run.filename+'/PCR_comonents_'+str(x_com)+'_and_'+str(y_com)
		tempfig.savefig(plotFileName.replace('.','p')+ui['file_extension'])
		temp_cbar.remove()
		tempax.cla()
	return
#Function for plotting the feature importance of features in RandomForestRegressor
def plot_feature_importance(event):
	run=PLSR.run
	if not run.ui['reg_type'] == 'Tree':
		return
	"""Function for plotting the feature importance of features in the Random
	Forest Regressor. Feature importance is shown in a plot overlaying the
	data plot. The feature importance is also saved in a separate plot if the
	"Save" option is selected."""
	feature_importance=run.last_reg_module.regr.feature_importances_
	common_variables=run.common_variables
	wavenum=run.last_complete_case.wavenumbers
	ui=run.ui
	ui['fig_per_row']=int(run.frame.buttons['fig_per_row'].get())
	ui['max_plots']=int(run.frame.buttons['max_plots'].get())
	ax=PLSRsave.get_or_make_absorbance_ax(run)
	PLSRsave.add_feature_importance_twinx(ax,common_variables,ui,xax,feature_importance)
	ax=fns.add_axis(common_variables.fig,ui['fig_per_row'],ui['max_plots'])
	PLSRsave.plot_feature_importance(ax,common_variables,ui,xax,feature_importance)
	if ui['save_check_var']:
		tempax=common_variables.tempax
		tempfig=common_variables.tempfig
		common_variables.tempfig.subplots_adjust(bottom=0.13,left=0.15, right=0.85, top=0.97)
		PLSRsave.PlotAbsorbance(tempax,tempfig,run.last_complete_case.active_wavenumers,ui,wavenumbers,common_variables.original_T.X,common_variables.original_V.X)
		twinx=PLSRsave.add_feature_importance_twinx(tempax,common_variables,run.ui,xax,feature_importance)
		plotFileName=run.filename+'/transmissionFullAndFeatureImportance'
		tempfig.savefig(plotFileName.replace('.','p')+ui['file_extension'])
		tempax.cla()
		twinx.remove()
		common_variables.tempfig.subplots_adjust(bottom=0.13,left=0.15, right=0.97, top=0.97)
		PLSRsave.plot_feature_importance(tempax,common_variables,ui,xax,feature_importance)
		plotFileName=run.filename+'/FeatureImportance'
		tempfig.savefig(plotFileName.replace('.','p')+ui['file_extension'])
		tempax.cla()
	run.draw()

def stability_selection(event):
	run=PLSR.run
	run.last_reg_module.neural_net.do_stability_selection(run)

def plot_node_correlations(event):
	run=PLSR.run
	if not run.ui['reg_type'] == 'NeuralNet':
		return
	print(run.complete_cases[-1].keywords['NN_type'])
	if run.complete_cases[-1].keywords['NN_type']=='Convolutional':
		return stability_selection(event)
	return
	if not hasattr(run,'last_Xval_case'):
		print('Not done running')
		return
	ui=run.ui
	ui['fig_per_row']=int(run.frame.buttons['fig_per_row'].get())
	ui['max_plots']=int(run.frame.buttons['max_plots'].get())
	ax=fns.add_axis(run.common_variables.fig,ui['fig_per_row'],ui['max_plots'])
	ax.plot_type='NN node map'
	V=run.last_Xval_case.V
	if len(V.X)==0:
		V=run.last_Xval_case.T

	transformedDataset=run.last_reg_module.scaler.transform(V.X)
	values=run.last_reg_module.neural_net.get_values(transformedDataset)
	run.NNvalues=values
	y_midpoint=run.last_reg_module.neural_net.layer_size/2
	X=[]
	Y=[]
	corr=[]
	y_rot=np.rot90(np.atleast_2d(V.Y),-1)
	corr_param=run.last_reg_module.neural_net.y_scaler.transform(y_rot).reshape(-1)
	for j in range((len(values)+2)//3):
		layer=values[j*3]
		shape=layer.shape
		corr.append([])
		for i in range(shape[1]):
			slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(layer[:,i], corr_param)
			corr[-1].append(r_value**2)
			X.append(j)
			Y.append(i)
	run.corr=corr
	flat_corr=[item for sublist in corr for item in sublist]
	sc=ax.scatter(X,Y,c=flat_corr,cmap='viridis')
	cbar=run.fig.colorbar(sc, ax=ax)
	cbar.set_label(r'r$^2$')
	run.draw()

def plot_node_activation_vector(event):
	run=PLSR.run
	if not run.ui['reg_type'] == 'NeuralNet':
		return
	ui=run.ui
	layer=int(event.xdata+0.5)
	node=int(event.ydata+0.5)
	try:
		r2=run.corr[layer][node]
		print('node at x',layer,'y',node,'r**2',r2)
	except:
		print('no node at x',layer,'y',node)
		return

	ui['fig_per_row']=int(run.frame.buttons['fig_per_row'].get())
	ui['max_plots']=int(run.frame.buttons['max_plots'].get())
	ax=fns.add_axis(run.common_variables.fig,ui['fig_per_row'],ui['max_plots'])
	weights=run.last_reg_module.neural_net.get_weights()

	V=run.last_Xval_case.V
	if len(V.X)==0:
		V=run.last_Xval_case.T
	data=run.last_reg_module.scaler.transform(V.X)
	input_dim = data.shape[1]
	num_inputs = data.shape[0]

	print(data.shape)
	sensitivity=[[]]
	for i in range(input_dim):
		sensitivity[-1].append(np.zeros(data.shape))
		sensitivity[-1][-1][:,i]=1
	activation = run.last_reg_module.neural_net.activation
	for ll in range((len(weights)+2)//3):
		sensitivity.append([])
		for i in range(input_dim):
			sensitivity[-1].append(sensitivity[-2][i] @ weights[ll*3][0][:])
			#sensitivity[-1][-1] += weights[ll*3][1] #do not include this, we are calculating derivatives, not response
		data = data @ weights[ll*3][0][:]
		data += weights[ll*3][1] #add bias
		if not ll==(len(weights)-1)//3:
			for i in range(input_dim):
				sensitivity[-1][i] = activation(sensitivity[-1][i],pivot=data)
				#print(np.sum(sensitivity[-1][-1]==0))
			data = activation(data)
			#for i in range(data.shape[1]):
			#	print(np.sum(data[:,i]==0))
	#sensitivity[layer][input][wavenumber,node
	#sensitivity[layer+1]=np.array(sensitivity[layer+1])
	#ode_sensitivity=sensitivity[layer+1][:,:,node]
	run.sensitivity=sensitivity
	node_sensitivity=[]
	for i in range(input_dim):
		node_sensitivity.append(sensitivity[layer+1][i][:,node])
	node_sensitivity=np.array(node_sensitivity)
	#print(len(sensitivity),sensitivity[layer+1].shape,node_sensitivity.shape)
	sense_vector=(np.average(node_sensitivity,axis=1))
	sense_std=(np.std(node_sensitivity,axis=1))
	#for s in sensitivity[-1]:
	#print(run.sense_vector[layer+1][node])
	wavenum=run.last_complete_case.wavenumbers
	wavenum=wavenum[run.last_complete_case.active_wavenumers]
	ax.errorbar(wavenum,sense_vector,yerr=sense_std,color=[1,0,0,1],ecolor=[0,0,1,1])
	ax.invert_xaxis()
	#ax.plot(wavenum,sense_vector)
	#ax=fns.add_axis(run.common_variables.fig,ui['fig_per_row'],ui['max_plots'])
	#ax.plot(wavenum,sense_std)
	'''ax=fns.add_axis(run.common_variables.fig,ui['fig_per_row'],ui['max_plots'])
	ax.plot(wavenum,node_sensitivity,'+')
	ax.invert_xaxis()
	ax=fns.add_axis(run.common_variables.fig,ui['fig_per_row'],ui['max_plots'])
	ax.plot(wavenum,run.last_reg_module.scaler.transform(V.X).swapaxes(0,1),'+')
	ax.invert_xaxis()'''

	'''ax=fns.add_axis(run.common_variables.fig,ui['fig_per_row'],ui['max_plots'])
	data=run.last_reg_module.neural_net.y_scaler.inverse_transform(data)
	ax.plot(V.Y,data,'o')'''
	#ax.plot(sense_std)

	run.draw()
	#ax.plot()

	#values=run.last_reg_module.neural_net.get_values(transformedDataset)

	'''inv_act=run.last_reg_module.neural_net.inv_activation
	out0=np.min(run.NNvalues[layer*3][:,node])
	out1=np.max(run.NNvalues[layer*3][:,node])
	#out0=inv_act(out0) # we collect vaue before the activation function, do not need this
	#out1=inv_act(out1) # we collect vaue before the activation function, do not need this
	out0-=weights[layer*3][1][node] #subtract bias
	out1-=weights[layer*3][1][node] #subtract bias
	out0=weights[layer*3][0][:,node]*out0 #multiply with connection to previous layer
	out1=weights[layer*3][0][:,node]*out1 #multiply with connection to previous layer
	out0=out0.reshape(-1,1)
	out1=out1.reshape(-1,1)
	for i in range(layer):
		s=len(out0)
		lay=layer-i-1
		out0=inv_act(out0) # inverse activation
		out1=inv_act(out1) # inverse activation
		out0-=weights[(lay)*3][1].reshape(-1,1) # subtract bias
		out1-=weights[(lay)*3][1].reshape(-1,1)
		out0=np.array(weights[lay*3][0]) @ out0.reshape(-1,1) # multiply with connection to previous layer
		out1=np.array(weights[lay*3][0]) @ out1.reshape(-1,1)
		#out0=out0*s/len(out0)
		#out1=out1*s/len(out1)
	out0=run.last_reg_module.scaler.inverse_transform(out0.reshape(-1))
	out1=run.last_reg_module.scaler.inverse_transform(out1.reshape(-1))
	ax.plot(out0)
	ax.plot(out1)
	run.draw()'''
	#weights[layer*2][0][:,node]
