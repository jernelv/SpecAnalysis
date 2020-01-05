import numpy as np
def get_files(list_of_files,max_range):
    list_of_measurements=[]
    regressionCurControlTypes=[]
    for file in list_of_files:
        temp=PLSRfile(file, max_range,list_of_measurements,regressionCurControlTypes)
    X=np.array([measurement.y for measurement in list_of_measurements]) #member of frame, so that they are saved for when calibration is run
    Y=np.array([measurement.trueValues for measurement in list_of_measurements])
    file_paths=[measurement.location for measurement in list_of_measurements]
    wavenumbers=list_of_measurements[0].x
    return X, Y, file_paths, wavenumbers, regressionCurControlTypes

class PLSRfile(object):
    ################################################################################################
    ############################### Function for loading files #####################################
    ################################################################################################
    def __init__ (self,location,max_range,list_of_measurements,regressionCurControlTypes,values=[]):
        self.max_range=max_range
        self.location=location
        self.values=values
        if len(self.values)>0:
            if self.values[0]>self.max_range:
                return
        if self.location[-4:len(self.location)]=='.dpt' or self.location[-4:len(self.location)]=='.txt' or self.location[-4:len(self.location)]=='.DPT':
            self.load_dpt()
            list_of_measurements.append(self)
        elif self.location[-6:len(self.location)]=='.laser':
            self.load_laser()
            list_of_measurements.append(self)
            return
        elif self.location[-5:len(self.location)]=='.list':
            with open(self.location) as f:
                for line in f:
                    splitline=list(filter(None, line.strip().split('\t'))) #this joins multiple delimiters by filtering empty lists
                    if splitline[0][0]=='#':
                        continue
                    if 'ilepath' in line:
                        for i, contrltytpe in enumerate(splitline[1:]):
                            if contrltytpe not in regressionCurControlTypes:
                                regressionCurControlTypes.append(contrltytpe)
                        continue
                    elif len(splitline)>0 and not line[0]=='#':
                        truevalues=[]
                        for trueval in splitline[1:]:
                            truevalues.append(float(trueval))
                        path='/'.join(self.location.split('/')[0:-1])+'/'+splitline[0]
                        temp=PLSRfile(path,self.max_range,list_of_measurements,regressionCurControlTypes,values=truevalues)
    def load_dpt(self):
        self.x=[]
        self.y=[]
        with open(self.location) as f:
            first_line = f.readline().strip()
            if ',' in first_line:
                for line in f:
                    x1,x2 = map(float,line.split(','))
                    self.x.append(x1)
                    self.y.append(x2)
            else:
                for line in f:
                    self.x.append(float(line.split()[0].strip()))
                    self.y.append(float(line.split()[1].strip()))
        self.trueValues=self.values
        self.x=np.array(self.x)
        self.y=np.array(self.y)

    def load_laser(self):
        self.x=[]
        self.y=[]
        with open(self.location) as f:
            for line in f:
                if not '#' in line:
                    self.x.append(float(line.split()[0].strip()))
                    self.y.append(float(line.split()[1].strip()))
        self.trueValues=self.values
        self.x=np.array(self.x)
        self.y=np.array(self.y)
