
import numpy as np
import copy
import fns
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from . import PLSRsave
from . import PLSRwavelengthSelection
from . import PLSRregressionMethods


class sequentialFeatureSelector():
	def __init__(self,common_variables,ui,case,draw_fun):
		self.case=case
		self.MLR_reg_module = PLSRregressionMethods.getRegModule('MLR',case.keywords,ui['scaling'],ui['mean_centering'])
		self.reg_module = PLSRregressionMethods.getRegModule(ui['reg_type'],case.keywords,ui['scaling'],ui['mean_centering'])
		self.ui=ui
		#self.independentVariables=con.GAIndependentVariables
		'''self.numberOfIndividuals=ui['GA_number_of_individuals']#100#
		if self.numberOfIndividuals//2==0:
			print('Number of individuals must be odd, as they are mated in pairs, and the best is always kept. Changed to: '+str(ui['GA_number_of_individuals']+1))'''
		self.common_variables=common_variables

		self.T=copy.deepcopy(case.T)
		self.V=copy.deepcopy(case.V)
		#cut away excess datapoints as described by user
		self.numDatapoints=self.T.X.shape[1]
		self.draw_fun=draw_fun
		if self.reg_module.type=='regression':
			if ui['WS_loss_type']=='X-validation on training':
				self.rmse_string='RMSECV'
			elif ui['WS_loss_type']=='RMSEC on training':
				self.rmse_string='RMSEC'
			else: # ui['WS_loss_type']=='RMSEP on validation':
				self.rmse_string='RMSEP'
		else: #self.reg_module.type=='classifier':
			if ui['WS_loss_type']=='X-validation on training':
				self.rmse_string='CV % wrong'
			elif ui['WS_loss_type']=='RMSEC on training':
				self.rmse_string='calib % wrong'
			else: # ui['WS_loss_type']=='RMSEP on validation':
				self.rmse_string='pred % wrong'

	def run(self):
		'''if self.ui['SFS type']=='Forward':
			return self.forward_selection()
		if self.ui['SFS type']=='Backwards':
			return self.backwards_selection()
			def forward_selection(self):'''
		wavenumbers=self.case.wavenumbers
		if 'Forward' in self.ui['SFS type']:
			direction='Forward '
			current_active_wavenumbers=np.zeros(len(wavenumbers), dtype=bool)
		elif 'Backward' in self.ui['SFS type']:
			direction='Backward'
			current_active_wavenumbers=np.ones(len(wavenumbers), dtype=bool)
		if self.ui['SFS_floating']:
			floating=True
		else:
			floating=False
		ax=fns.add_axis(self.common_variables.fig,self.ui['fig_per_row'],self.ui['max_plots'])
		# calculate the needed X-val splits and store them
		PLSRwavelengthSelection.WS_getCrossvalSplits([0,1],self.T,self.V,self.ui,use_stored=False)
		PLSRsave.PlotChromosomes(ax,wavenumbers,[],self.ui,ylabel='Iteration')
		if self.ui['SFS type']=='Forward':
			current_active_wavenumbers=np.zeros(len(wavenumbers), dtype=bool)
		elif self.ui['SFS type']=='Backwards':
			current_active_wavenumbers=np.ones(len(wavenumbers), dtype=bool)
		best_historic_active=[]
		best_loss=[]
		generation=0
		while True:
			#main step
			if direction=='Forward ':
				trail_active_wavenumbers=self.get_trails_forward(current_active_wavenumbers)
			else: # direction=='Backward'
				trail_active_wavenumbers=self.get_trails_backward(current_active_wavenumbers)
			if len(trail_active_wavenumbers)==0:
				break
			trail_active_wavenumbers=cut_previous(trail_active_wavenumbers,best_historic_active)
			current_active_wavenumbers, l, out_str= self.do_pass(trail_active_wavenumbers,generation)
			print(direction+' '+out_str)
			best_loss.append(l)
			PLSRsave.PlotChromosome(ax,wavenumbers,current_active_wavenumbers,generation)
			self.draw_fun()
			best_historic_active.append(copy.copy(current_active_wavenumbers))
			best_historic_generation=np.argmin(best_loss)
			generation+=1
			if generation==self.ui['SFS_max_iterations']:
				break
			if floating:
				while True:
					if direction=='Forward ':
						if np.sum(current_active_wavenumbers)==1:
							break
						else:
							trail_active_wavenumbers=self.get_trails_backward(current_active_wavenumbers) #reverse of main loop
					else: # direction=='Backward'
						if np.sum(current_active_wavenumbers)==len(current_active_wavenumbers):
							break
						trail_active_wavenumbers=self.get_trails_forward(current_active_wavenumbers) #reverse of main loop
					trail_active_wavenumbers=cut_previous(trail_active_wavenumbers,best_historic_active)
					if len(trail_active_wavenumbers)==0:
						break
					best_trail, l, out_str = self.do_pass(trail_active_wavenumbers,generation)
					if l<best_loss[-1]:
						print('Floating'+' '+out_str)
						current_active_wavenumbers=best_trail
						best_loss.append(l)
						PLSRsave.PlotChromosome(ax,wavenumbers,current_active_wavenumbers,generation)
						self.draw_fun()
						best_historic_active.append(copy.copy(current_active_wavenumbers))
						best_historic_generation=np.argmin(best_loss)
						generation+=1
					else:
						break
			if generation==self.ui['SFS_max_iterations'] or best_historic_generation<len(best_loss)-self.ui['SFS_num_after_min'] or np.sum(current_active_wavenumbers)==self.ui['SFS_target']:
				break


		print('best iteration '+str(best_historic_generation+1)+', best '+self.rmse_string+'  = '+PLSRsave.custom_round(best_loss[best_historic_generation],2))
		PLSRsave.PlotChromosome(ax,wavenumbers,best_historic_active[best_historic_generation],best_historic_generation,color=[1,0,0,1])


		if self.ui['save_check_var']==1:
			PLSRsave.PlotChromosomes(self.common_variables.tempax,wavenumbers,best_historic_active,self.ui,ylabel='Iteration')
			PLSRsave.PlotChromosome(self.common_variables.tempax,wavenumbers,best_historic_active[best_historic_generation],best_historic_generation,color=[1,0,0,1])
			self.common_variables.tempfig.subplots_adjust(bottom=0.13,left=0.15, right=0.97, top=0.9)
			unique_keywords=PLSRsave.get_unique_keywords_formatted(self.common_variables.keyword_lists,self.case.keywords)
			plotFileName=case.folder+self.ui['reg_type']+unique_keywords.replace('.','p')+'SFS'
			self.common_variables.tempfig.savefig(plotFileName.replace('.','p')+self.ui['file_extension'])
		return best_historic_active[best_historic_generation]

	def get_trails_forward(self,current_active_wavenumbers):
		trail_active_wavenumbers=[]
		for i,act in enumerate(current_active_wavenumbers):
			if act==False:
				trail_active_wavenumbers.append(copy.copy(current_active_wavenumbers))
				trail_active_wavenumbers[-1][i]=True
		return trail_active_wavenumbers

	def get_trails_backward(self,current_active_wavenumbers):
		trail_active_wavenumbers=[]
		for i,act in enumerate(current_active_wavenumbers):
			if act==True:
				trail_active_wavenumbers.append(copy.copy(current_active_wavenumbers))
				trail_active_wavenumbers[-1][i]=False
		return trail_active_wavenumbers

	def do_pass(self,trail_active_wavenumbers,generation):
		losses,used_mlr=PLSRwavelengthSelection.WS_evaluate_chromosomes(self.reg_module,
				self.T, self.V, trail_active_wavenumbers,
				use_stored=True, backup_reg_module=self.MLR_reg_module)
		best_i=np.argmin(losses)
		if used_mlr:
			out_str='iteration '+str(generation+1)+' done, best '+self.rmse_string+' = '+PLSRsave.custom_round(losses[best_i],2)+' (using MLR)'
			return trail_active_wavenumbers[best_i],losses[best_i]+10000,out_str # increase the value of the loss here, because we do not wat to permit the algorithm for choosing this as the best case
		else:
			out_str='iteration '+str(generation+1)+' done, best '+self.rmse_string+' = '+PLSRsave.custom_round(losses[best_i],2)
			return trail_active_wavenumbers[best_i],losses[best_i],out_str

def cut_previous(trail_active_wavenumbers,best_historic_active):
	new_trails=[]
	for trail in trail_active_wavenumbers:
		keep=True
		for historic in best_historic_active:
			if (trail==historic).all():
				keep=False
				break
		if keep:
			new_trails.append(trail)
	return new_trails
