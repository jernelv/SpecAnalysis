import numpy as np
import fns
import sys

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
class moduleClass:
	filetypes = ['bin']
	def __init__(self, fig, locations, frame, ui):
		self.ui=ui
		self.fig=fig
		self.frame=frame
		fname=locations[0]
		StartWL=1200
		EndWL=925
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
		self.pulseIntensity=pulseIntensity[0:i+1]
		self.pulseIntensity=self.pulseIntensity/np.average(self.pulseIntensity)
		self.pulseIntensity[:] = [x-1 for x in self.pulseIntensity]
		return
	def run(self):
		self.fig.clf()
		ax=fns.add_axis(self.fig,1)
		numPulses=100
		#self.average_pulses=[]
		self.ms=[255]
		for i, n in enumerate(self.ms):
			average_2_pulses=averageN(self.pulseIntensity,n)
		self.plot_me(ax, numPulses)

		if self.frame.save_check_var.get():
			tempfig = self.frame.hidden_figure
			tempfig.set_size_inches(4*1.2, 3*1.2)
			tempfig.set_dpi(300)
			tempfig.clf()
			tempax = tempfig.add_subplot(1, 1, 1)
			tempfig.subplots_adjust(bottom=0.17,left=0.16, right=0.97, top=0.97)
			self.plot_me(tempax,numPulses)

			filename=self.frame.name_field_string.get()
			tempfig.savefig(filename+'.png')
			tempfig.savefig(filename+'.svg')
		#ax2=fns.add_axis(self.fig,1)
		#average_2_pulses=averageN(self.pulseIntensity,8)
		#ax2.scatter(range(len(average_2_pulses)),average_2_pulses)
		#if self.ui['save_check']:
		#	self.fig.savefig(self.ui['save_filename']+'.png')
		return
	def plot_me(self,ax,numPulses):

		figure=ax.figure
		#[x,y,width,height]
		pos = [0.5, 0.25, 0.45, 0.3]
		newax = figure.add_axes(pos)
		for i,n in enumerate(self.ms):
			average_2_pulses=averageN(self.pulseIntensity,n)
			xax=np.array(range(numPulses))+numPulses*i
			ax.scatter(xax,100*average_2_pulses[1024:numPulses*200+1024:200],3)
			newax.loglog(n,np.std(100*average_2_pulses[1024:numPulses*200+1024:200]),'x')
			#ax.text(i*numPulses+numPulses/2,0.0075,str(n))
			#ax.text(i+0.5,1.075,str(m))

		ax.set_xticks((np.arange(len(self.ms))+0.5)*numPulses)
		ax.set_xticklabels(self.ms)
		#ax.invert_xaxis()
		ax.set_xlabel(r'Number of averaged laser pulses')
		ax.set_ylabel(r'Deviation from mean pulse intensity [%]')

		return
	def addButtons():
		buttons = [
		#{'key': 'joinPNG_orientation', 'type': 'radio:text', 'texts': ['horizontal', 'vertical'], 'tab': 0, 'row': 0} ,
		]
		return buttons
