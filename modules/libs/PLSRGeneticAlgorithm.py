
import numpy as np
import copy
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from . import PLSRsave
from . import PLSRwavelengthSelection
from . import PLSRregressionMethods


class GeneticAlgorithm():
	def __init__(self,common_variables,ui,case):
		self.case=case
		self.reg_module=PLSRregressionMethods.getRegModule(ui['reg_type'],case.keywords)
		self.ui=ui
		#self.independentVariables=con.GAIndependentVariables
		self.numberOfIndividuals=ui['GA_number_of_individuals']#100#
		if self.numberOfIndividuals//2==0:
			print('Number of individuals must be odd, as they are mated in pairs, and the best is always kept. Changed to: '+str(ui['GA_number_of_individuals']+1))
		self.common_variables=common_variables

		self.T=copy.deepcopy(case.T)
		self.V=copy.deepcopy(case.V)
		#self.V=copy.deepcopy(V) not used
		#cut away excess datapoints as described by user
		self.numDatapoints=self.T.X.shape[1]

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

	def run(self,ax,wavenumbers,folder,draw_fun):
		# calculate the needed X-val splits and store them
		PLSRwavelengthSelection.WS_getCrossvalSplits([0,1],self.T,self.V,self.ui,use_stored=False)
		#prepare plot
		PLSRsave.PlotChromosomes(ax,wavenumbers,[],self.ui)
		ax.set_ylim([self.ui['GA_max_number_of_generations']+0.5,-0.5])
		# create initial population
		self.makeInitialPopulation()
		# evaluate initial populatoin
		self.evaluatePopulation()
		#sort population
		self.sortPopulation()
		PLSRsave.PlotChromosome(ax,wavenumbers,self.population[0],0)
		#for num generations:
		bestAfterGeneration=[self.population[0]]
		for generation in range(self.ui['GA_max_number_of_generations']):
			#select indviduals from population
			self.selectParents()
			#perform Crossover
			self.makeCrossoverChildren()
			#perform Mutation
			self.mutate()
			#add in best from previous gen
			self.nextGen=np.concatenate([self.nextGen,[self.population[0]]])
			#the children shal ideanherit the earth!
			self.population=self.nextGen
			#evaluate this generation of individuals
			self.evaluatePopulation()
			#sort population
			self.sortPopulation()
			bestAfterGeneration.append(self.population[0])
			print('gen '+str(generation+1)+' done, min '+self.rmse_string+' = '+PLSRsave.custom_round(self.RMSEPs[0],2))
			draw_fun()
			PLSRsave.PlotChromosome(ax,wavenumbers,self.population[0],generation+1)
			draw_fun()
		bestDatapoints=self.population[0]
		if self.ui['save_check_var']==1:
			PLSRsave.PlotChromosomes(self.common_variables.tempax,wavenumbers,bestAfterGeneration,self.ui)
			self.common_variables.tempfig.subplots_adjust(bottom=0.13,left=0.15, right=0.97, top=0.9)
			unique_keywords=PLSRsave.get_unique_keywords_formatted(self.common_variables.keyword_lists,self.case.keywords)
			plotFileName=folder+self.ui['reg_type']+unique_keywords.replace('.','p')+'_genetic_algorithm'
			self.common_variables.tempfig.savefig(plotFileName.replace('.','p')+self.ui['file_extension'])
		return bestDatapoints

	def makeInitialPopulation(self):
		self.population=np.random.randint(2,size=(self.numberOfIndividuals,self.numDatapoints))
		self.population=np.array(self.population, dtype=bool)
		#self.population=np.random.randint(self.numDatapoints,size=(self.numberOfIndividuals,self.independentVariables))

	def evaluatePopulation(self):
		self.RMSEPs, _ = PLSRwavelengthSelection.WS_evaluate_chromosomes(self.reg_module,
				self.T, self.V, self.population,
				use_stored=True)


	def sortPopulation(self):
		argsort = self.RMSEPs.argsort()
		self.RMSEPs = self.RMSEPs[argsort]
		self.population[:,:] = self.population[argsort,:]

	def selectParents(self):
		#binary tournament selection
		numParents = round(len(self.population-0.1)/2.0)*2 #must be even, crossover requires 2 parents, if odd will round down
		self.parents = []
		champion1 = np.random.choice(len(self.population), numParents)
		champion2 = np.random.choice(len(self.population), numParents)
		for c1, c2 in zip(champion1,champion2):
			if self.RMSEPs[c1]>self.RMSEPs[c2]:
				self.parents.append(self.population[c2])
			else:
				self.parents.append(self.population[c1])
		self.parents = np.array(self.parents)
		'''# Fitness proportionate selection implemented below, but not used, also known as roulette wheel selection. Fitness linear from 2(best) to 0(worst)
		#self.ratings sorted from 2 to 0 where the first element has the lowest RMS
		#rangestep=2.0/(len(self.population)-1)
		#self.ratings=np.flip(np.arange(0,2+0.5*rangestep,rangestep))
		#self.insertionRate : variable between 0 and 1
		numParents=round(len(self.population-0.1)/2.0)*2 #must be even, crossover requires 2 parents, if odd will round down
		normalizedRatings=self.ratings/sum(self.ratings)
	 	parentsIndexes = np.random.choice(len(self.population), numParents, p=normalizedRatings) #this is with replacement, i.e. one parent can be included multiple timesself
		self.parents = self.population[parentsIndexes]
		# an alternative is to use Stochastic universal sampling, but this is not implemented'''

	def makeCrossoverChildren(self):
		#self.parents=np.random.shuffle(self.parents) # shuffle not required as array is allready shuffled
		P1=self.parents[0:int(len(self.parents))//2]
		P2=self.parents[int(len(self.parents))//2:]
		xoverpoints=np.random.randint(self.numDatapoints, size=len(P1))

		self.nextGen=[]
		for p1, p2, xoverpoint in zip(P1,P2,xoverpoints):
			if np.random.rand() > self.ui['GA_crossover_rate']:
				self.nextGen.append(p1)
				self.nextGen.append(p2)
			else:
				self.nextGen.append(np.concatenate([p1[0:xoverpoint],p2[xoverpoint:]]))
				self.nextGen.append(np.concatenate([p2[0:xoverpoint],p1[xoverpoint:]]))

	def mutate(self):
		#self.nextGen=np.array(self.nextGen)
		mutationMatrix=np.random.choice(2, size=[len(self.nextGen),len(self.nextGen[0])],p=[1-self.ui['GA_mutation_rate'],self.ui['GA_mutation_rate']])
		self.nextGen=np.logical_xor(self.nextGen,mutationMatrix)
