
from tkinter import ttk
import tkinter
class turn_onoff_modules:
    def __init__(self,parent_frame):
        self.superframe = ttk.Frame(parent_frame, relief=tkinter.RAISED, borderwidth=0)
        self.superframe.pack(side=tkinter.TOP,fill=tkinter.BOTH, expand=0)
        self.childframes=[]
        self.numbuttons=0
        self.modulemap={}

    def make_or_get_modulebutton(self,name):
        if not name in self.modulemap:
            if self.numbuttons%4==0:
                self.addrow()
            self.modulemap[name] = tkinter.IntVar()
            self.modulemap[name].set(1)
            self.module_reload = ttk.Checkbutton(self.childframes[-1] , text=name, variable=self.modulemap[name])
            self.module_reload.pack(side=tkinter.LEFT)
            self.numbuttons+=1
            return 1
        else:
            return self.modulemap[name].get()
    def addrow(self):
            self.childframes.append(ttk.Frame(self.superframe, relief=tkinter.RAISED, borderwidth=0))
            self.childframes[-1].pack(side=tkinter.TOP,fill=tkinter.BOTH, expand=0)
