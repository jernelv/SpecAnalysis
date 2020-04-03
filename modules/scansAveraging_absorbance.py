import numpy as np
import fns
import copy
from .libs import signal_alignment
from .libs import signal_alignment

#function for averaging over n pulses
def averageN(y,n=5):
	nh=int(n/2)
	rest=n-2*nh
	y2=np.array(y.copy())
	l=len(y)
	y2[int(nh):l-1-nh]=0.0
	for i in range(-nh,nh+rest,1):
		#print(int(nh)+i,l-1-nh+i,l)
		y2[int(nh):l-1-nh]+=y[int(nh)+i:l-1-nh+i]/float(n)
	return y2
def averageM(y,m=5):
	for j in range(y.shape[1]):
		y2=np.concatenate([copy.copy(y[:,j]),copy.copy(y[:,j])])
		for i in range(y.shape[0]):
			y[i,j]=np.average(y2[i:i+m])
	return y
class moduleClass:
	filetypes = ['bin']
	def __init__(self, fig, locations, frame, ui):
		self.ui=ui
		self.fig=fig
		self.scans=[]
		self.frame=frame
		for fname in locations:
			data=np.fromfile(fname,dtype=np.int16)

			rate=80000000 #only valid for 100kHz repetition rate
			####################### get periodicity in data with FFT
			fftfreq=np.fft.fftfreq(24000)[1:]
			fft=np.fft.fft(data[0:24000])[1:]/fftfreq # divide by fftfreq
			period=int(1/fftfreq[np.argmax(abs(fft))])
			period=800
			#print(period)

			####################### Get first pulse
			somezeroes = np.zeros(100, dtype=np.int16)
			data = np.concatenate((somezeroes, data)) ##adding some zeroes to the beginning of data
			#bin1 = np.concatenate((somezeroes, bin1)) ##adding some zeroes to the beginning of data
			#bin2 = np.concatenate((somezeroes, bin2)) ##adding some zeroes to the beginning of data
			maxstart=max(data[0:2000])
			nextpulse=np.argmax(data[0:2000]-np.arange(0,maxstart,maxstart/(2000.0-0.5)))
			pulsewindow=50
			correctPlace=data[nextpulse-pulsewindow:nextpulse+pulsewindow]
			#nextpulse+=np.argmax(correctPlace)-40
			corvector=np.arange(-pulsewindow,pulsewindow,1)

			nextpulse+=int(np.sum(correctPlace*corvector)/np.sum(correctPlace))

			####################### get max value of each pulse
			numpulses=int(len(data)/period)+100 #make array extra long, make sure it is long enough
			pulseIntensity=np.zeros(numpulses) #np.zeros(len(data)/period)
			dichal1=np.zeros(numpulses) #np.zeros(len(data)/period)
			i=0
			while nextpulse+20 < len(data) and nextpulse < 80000000:
				#print(nextpulse,i)
				if i%500==0:
					# every 500 pulses: refine pulse position
					#plt.plot(data[nextpulse-20:nextpulse+20])
					correctPlace=data[nextpulse-pulsewindow:nextpulse+pulsewindow]
					#nextpulse+=np.argmax(correctPlace)-80
					#print(np.sum(correctPlace))
					if np.sum(correctPlace) >1000:
						nextpulse+=int(np.sum(correctPlace*corvector)/np.sum(correctPlace))
				#pulseIntensity[i]=np.max(data[nextpulse-40:nextpulse+40])
				pulseIntensity[i]=np.sum(data[nextpulse-pulsewindow:nextpulse+pulsewindow]) # integrate the pulse
				#dichal1[i]=bin1[nextpulse]
				nextpulse+=period
				i+=1
			####################### cut off excess length of pulseIntensity
			i=-1
			while pulseIntensity[i]==0:
				i-=1
			numpulses=numpulses+i
			pulseIntensity=pulseIntensity[0:i+1]
			self.scans.append(pulseIntensity)
		#self.scans=np.array(self.scans)
		return
	def run(self):
		self.fig.clf()
		ax=fns.add_axis(self.fig,1)
		StartWL=1200
		EndWL=925
		minscanlength=np.inf
		for scan in self.scans:
			if len(scan)<minscanlength:
				minscanlength=len(scan)
		for i,scan in enumerate(self.scans):
			self.scans[i]=scan[0:minscanlength]
			n=255
			self.scans[i]=averageN(self.scans[i],n)
		self.scans=np.array(self.scans)
		self.averagescans=np.average(self.scans,axis=0)

		for i,scan in enumerate(self.scans):
			self.scans[i]=np.log10(scan/self.averagescans)
		for i, scan in enumerate(self.scans):
			if i > 0:
				s = signal_alignment.chisqr_align(self.scans[0], scan, [0,20000], init=0, bound=50)
				print(s)
				self.scans[i] = signal_alignment.shift(scan, s, mode='nearest')
		#StartWL=1200
		#EndWL=925
		#self.wavenumbers=StartWL+(EndWL-StartWL)*np.arange(minscanlength)/minscanlength
		StartWL=1200
		EndWL=925
		self.wavenumbers=StartWL+(EndWL-StartWL)*np.arange(minscanlength)/minscanlength
		numPulses=1000
		step=100
		self.ms=[10]
		self.averaged_scans=[]
		for i,m in enumerate(self.ms):
			self.averaged_scans.append(copy.deepcopy(self.scans))
			self.averaged_scans[-1]=averageM(self.averaged_scans[-1],m)
		self.plot_me(ax,step,numPulses,EndWL,StartWL)

		if self.frame.save_check_var.get():
			tempfig = self.frame.hidden_figure
			tempfig.set_size_inches(4*1.2, 3*1.2)
			tempfig.set_dpi(300)
			tempfig.clf()
			tempax = tempfig.add_subplot(1, 1, 1)
			tempfig.subplots_adjust(bottom=0.17,left=0.16, right=0.97, top=0.97)
			self.plot_me(tempax,step,numPulses,EndWL,StartWL)

			filename=self.frame.name_field_string.get()
			tempfig.savefig(filename+'.png')
			tempfig.savefig(filename+'.svg')
		return

	def plot_me(self,ax,step,numPulses,EndWL,StartWL):

		figure=ax.figure
		#[x,y,width,height]
		pos = [0.3, 0.25, 0.3, 0.2]
		newax = figure.add_axes(pos)
		for i,m in enumerate(self.ms):
			dat=self.averaged_scans[i][:,step//2:step*numPulses+step//2:step].swapaxes(0,1)
			dat=np.std(dat,axis=1)
			#ax.semilogy(self.wavenumbers[step//2:step*numPulses+step//2:step],
			#	dat*100)
			xax=self.wavenumbers[step//2:step*numPulses+step//2:step]
			if m==1:
				label='1 scan'
			else:
				label=str(m)+' scans'
			'''ax.fill_between(xax[1:-1],
				-dat[1:-1]*100,
				dat[1:-1]*100,
				label=label)'''
			ax.plot(xax[1:-1],
				dat[1:-1],
				label=label)
			newax.loglog([m],[np.average(dat[1:-27000//100])],'x') #1200-1000
		ax.legend(loc=2)
			#ax.text(i+0.5,1.075,str(m))
		#ax.set_xticks(np.arange(len(self.ms))+0.5)
		#ax.set_xticklabels(self.ms)
		ax.invert_xaxis()
		#ax.set_xlabel(r'Wavenumbers [cm$^-1$]')
		ax.set_ylabel(r'Deviation from mean intensity [%]')
		ax.set_xlabel(r'Wavenumber [cm-1]')
		#ax.set_ylim([-1,1])
		return
	def addButtons():
		buttons = [
		#{'key': 'joinPNG_orientation', 'type': 'radio:text', 'texts': ['horizontal', 'vertical'], 'tab': 0, 'row': 0} ,
		]
		return buttons
