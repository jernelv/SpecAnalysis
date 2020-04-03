
import numpy as np
import copy
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut
from . import PLSRsave
from . import PLSRwavelengthSelection
from . import PLSRregressionMethods


class forwardSelection():
	def __init__(self,common_variables,ui,case):
		self.case=case
		self.MLR_regModule = PLSRregressionMethods.getRegModule('MLR',case.keywords)
		self.regModule = PLSRregressionMethods.getRegModule(ui['reg_type'],case.keywords)
		self.ui=ui
		#self.independentVariables=con.GAIndependentVariables
		'''self.numberOfIndividuals=ui['GA_number_of_individuals']#100#
		if self.numberOfIndividuals//2==0:
			print('Number of individuals must be odd, as they are mated in pairs, and the best is always kept. Changed to: '+str(ui['GA_number_of_individuals']+1))'''
		self.common_variables=common_variables

		self.T=copy.deepcopy(case.T)
		self.V=copy.deepcopy(case.V)
		#self.V=copy.deepcopy(V) not used
		#cut away excess datapoints as described by user
		self.numDatapoints=self.T.X.shape[1]
	def run(self,ax,wavenumbers,folder,draw_fun):
		PLSRsave.PlotChromosomes(ax,wavenumbers,[],self.ui,ylabel='Iteration')

		current_active_wavenumbers=np.zeros(len(wavenumbers), dtype=bool)
		best_historic_active=[]
		best_loss=[]
		generation=0
		while True:
			trail_active_wavenumbers=[]
			for i,act in enumerate(current_active_wavenumbers):
				if act==False:
					trail_active_wavenumbers.append(copy.copy(current_active_wavenumbers))
					trail_active_wavenumbers[-1][i]=True
			if len(trail_active_wavenumbers)==0:
				break
			losses=[]
			using_mlr=False
			for active_wavenumers in trail_active_wavenumbers:
				try:
					losses.append(PLSRwavelengthSelection.WS_getRMSEP(self.regModule,active_wavenumers,self.T,self.V,self.ui,use_stored=True))
				except:
					using_mlr=True
					losses.append(PLSRwavelengthSelection.WS_getRMSEP(self.MLR_regModule,active_wavenumers,self.T,self.V,self.ui,use_stored=True))
			best_i=np.argmin(losses)
			if using_mlr:
				print('iteration '+str(generation+1)+' done, best x-val RMSEP = '+PLSRsave.custom_round(losses[best_i],2)+' (using MLR)')
			else:
				print('iteration '+str(generation+1)+' done, best x-val RMSEP = '+PLSRsave.custom_round(losses[best_i],2))

			current_active_wavenumbers=trail_active_wavenumbers[best_i]
			PLSRsave.PlotChromosome(ax,wavenumbers,current_active_wavenumbers,generation)
			generation+=1
			draw_fun()
			best_loss.append(losses[best_i])
			best_historic_active.append(copy.copy(current_active_wavenumbers))
			best_historic_generation=np.argmin(best_loss)
			if best_historic_generation<len(best_loss)-20:
				break

		print('best iteration '+str(best_historic_generation+1)+', best x-val RMSEP = '+PLSRsave.custom_round(best_loss[best_historic_generation],2))
		PLSRsave.PlotChromosome(ax,wavenumbers,best_historic_active[best_historic_generation],best_historic_generation,color=[1,0,0,1])


		if self.ui['save_check_var']==1:
			PLSRsave.PlotChromosomes(self.common_variables.tempax,wavenumbers,best_historic_active,self.ui,ylabel='Iteration')
			PLSRsave.PlotChromosome(self.common_variables.tempax,wavenumbers,best_historic_active[best_historic_generation],best_historic_generation,color=[1,0,0,1])
			self.common_variables.tempfig.subplots_adjust(bottom=0.13,left=0.15, right=0.97, top=0.9)
			unique_keywords=PLSRsave.get_unique_keywords_formatted(self.common_variables.keyword_lists,self.case.keywords)
			plotFileName=folder+self.ui['reg_type']+unique_keywords.replace('.','p')+'_forward_selection'
			self.common_variables.tempfig.savefig(plotFileName.replace('.','p')+self.ui['file_extension'])
		return best_historic_active[best_historic_generation]
