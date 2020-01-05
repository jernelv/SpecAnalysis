import tkinter #from Tkinter import StringVar, Tk, Label, RIGHT, BOTH, RAISED, Text, TOP, BOTH, X, LEFT, W, N, E, S, Listbox, END,Radiobutton, IntVar, Checkbutton, DISABLED, NORMAL
from tkinter import ttk #from ttk import Frame, Button, Style, Label, Entry
import os
import numpy as np
class data_tree():
    def __init__(self, parent,data_tree_frame ):
        self.parent=parent
        ####################### place UI items
        self.data_tree_frame=data_tree_frame
        style = ttk.Style()
        #style.configure("mystyle.Treeview",font='TkFixedFont',size=9) #defined in navigator
        data_tree_frame_frame=ttk.Frame(data_tree_frame, relief=tkinter.RAISED, borderwidth=0)
        data_tree_frame_frame.pack(fill=tkinter.BOTH, expand=True,side=tkinter.TOP, anchor=tkinter.E)
        self.trvw=ttk.Treeview(data_tree_frame_frame,show="tree",style="mystyle.Treeview") # the navigator listbox
        self.monospace_font_widt=tkinter.font.Font(font='TkFixedFont', size=9).measure(' ')
        self.entrdct={}
        self.strdict={}
        self.strdict['']=''
        self.trvw.column('#0', stretch=0)
        self.trvw['columns'] = [1]
        self.trvw.column('#1',stretch=0)
        self.trvw.bind("<<TreeviewOpen>>", self.branch_open, "+")
        self.trvw.bind("<<TreeviewClose>>", self.branch_close, "+")
        self.trvw.bind("<Double-1>", self.OnDoubleClick)
        #self.trvw.bind("<<TreeviewSelect>>", self.trvw_select) # bind selecting elements to open close folder # passes the event to the function
        self.trvw.tag_configure('color0', background='#FFFFFF')
        self.trvw.tag_configure('color1', background='#7777FF')
        self.trvw.tag_configure('color2', background='#77FF77')
        self.trvw.tag_configure('color3', background='#FF7777')
        self.trvw.tag_configure('color12', background='#FFFF77')
        self.trvw.tag_configure('color23', background='#FF77FF')
        self.trvw.tag_configure('color13', background='#77FFFF')
        self.trvw.tag_configure('color123', background='#AAAAAA')
        self.trvw.pack(side=tkinter.LEFT,anchor=tkinter.S,fill=tkinter.BOTH, expand=1)
        vsb = ttk.Scrollbar(data_tree_frame_frame, orient="vertical", command=self.trvw.yview)
        vsb.pack(side=tkinter.LEFT,anchor=tkinter.S,fill=tkinter.BOTH, expand=0)
        self.trvw.configure(yscrollcommand=vsb.set)
        rebuildframe= ttk.Frame(data_tree_frame, relief=tkinter.RAISED, borderwidth=0)
        rebuildframe.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH)
        self.rebuildbutton=ttk.Button(rebuildframe, text="Rebuild", command=self.rebuild)
        self.rebuildbutton.pack(side=tkinter.LEFT,fill=tkinter.BOTH)
        self.simple_space_var = tkinter.IntVar()
        self.simple_space = ttk.Checkbutton(
            rebuildframe, text="full namespace", variable=self.simple_space_var)
        self.simple_space.pack(side=tkinter.LEFT,fill=tkinter.BOTH)
        self.max_children_string = tkinter.StringVar()
        self.max_children_string.set('20')
        self.max_children_field = ttk.Entry(
            rebuildframe, width=4, textvariable=self.max_children_string)
        self.max_children_field.pack(side=tkinter.LEFT)
        self.max_children_label = tkinter.Label(
            rebuildframe, text='children', background="#FFFFFF")
        self.max_children_label.pack(side=tkinter.LEFT)

        self.dict_or_dir=tkinter.IntVar()
        dict_button=tkinter.Radiobutton(rebuildframe,
                        text='__dict__',
                        # padx = 20,
                        # borderwidth=0,
                        indicatoron=0,
                        variable=self.dict_or_dir,
                        # command=self.Selectcolor,
                        value=0,
                        background="#FFFFFF")
        dict_button.pack(side=tkinter.LEFT)
        dir_button=tkinter.Radiobutton(rebuildframe,
                        text='__dir__',
                        # padx = 20,
                        # borderwidth=0,
                        indicatoron=0,
                        variable=self.dict_or_dir,
                        # command=self.Selectcolor,
                        value=1,
                        background="#FFFFFF")
        dir_button.pack(side=tkinter.LEFT)
        self.purge_buttons_button=ttk.Button(rebuildframe, text="purge buttons", command=parent.purge_buttons)
        self.purge_buttons_button.pack(side=tkinter.LEFT,fill=tkinter.BOTH)
        #self.parent.after(10, self.rebuild)
        #self.parent.after(10, self.get_widt)

    def OnDoubleClick(self, event):
        item = self.trvw.identify('item',event.x,event.y)
        self.parent.console_intput.insert("insert",self.strdict[item])
        return 'break'

    def add(self,parent_entry,entries,variables, strings):
        #entries -> list for left column
        #variables -> pointers to object -> list for right column
        try:
            entries,variables,strings=zip(*sorted(zip(entries, variables,strings)))
        except:
            None

        for entry,variable,string in zip(entries,variables,strings):
            done=0
            entry=str(entry)
            if isinstance(variable, str) or isinstance(variable, bytes)  :
                try:
                    values='str:\xa0'+str(variable)
                    if len(values)>25:
                        values=values[0:25]+'..'
                    self.trvw.insert(parent_entry, 'end',parent_entry+'%'+entry, text=entry,values=values.replace(' ','\xa0'))
                    done=1
                except:
                    None
            if done==0 and hasattr(variable,'shape') and not isinstance(variable, np.ndarray):
                try:
                    if not len(variable.shape)==0:
                        values='array, shape:'+str(variable.shape)
                        self.trvw.insert(parent_entry, 'end',parent_entry+'%'+entry, text=entry,values=values.replace(' ','\xa0'))
                        if hasattr(variable,'attrs') and hasattr(variable.attrs,'keys'):
                            self.trvw.insert(parent_entry+'%'+entry, 'end', text=' ') #dummy to make it openable
                    else:
                        try:
                            values='array, shape:'+str(variable.shape)+' '+format(variable, '.3e')
                        except:
                            values='array, shape:'+str(variable.shape)
                        self.trvw.insert(parent_entry, 'end',parent_entry+'%'+entry, text=entry,values=values.replace(' ','\xa0'))
                    done=1
                except:
                    None
            if done==0 and  isinstance(variable, dict):
                try:
                    values=str(type(variable))
                    if len(values)>25:
                        values=values[0:25]+'..'
                    self.trvw.insert(parent_entry, 'end',parent_entry+'%'+entry, text=entry,values=values.replace(' ','\xa0'),tags='color3')
                    self.trvw.insert(parent_entry+'%'+entry, 'end', text=' ') #dummy to make it openable
                    done=1
                except:
                    None
            if done==0 and  self.dict_or_dir.get()==0 and hasattr(variable,'__dict__'):
                try:
                    values=str(type(variable))
                    if len(values)>25:
                        values=values[0:25]+'..'
                    self.trvw.insert(parent_entry, 'end',parent_entry+'%'+entry, text=entry,values=values.replace(' ','\xa0'),tags='color1')
                    self.trvw.insert(parent_entry+'%'+entry, 'end', text=' ') #dummy to make it openable
                    done=1
                except:
                    None
            if done==0 and  self.dict_or_dir.get()==1 and hasattr(variable,'__dir__'):
                try:
                    values=str(type(variable))
                    if len(values)>25:
                        values=values[0:25]+'..'
                    self.trvw.insert(parent_entry, 'end',parent_entry+'%'+entry, text=entry,values=values.replace(' ','\xa0'),tags='color1')
                    self.trvw.insert(parent_entry+'%'+entry, 'end', text=' ') #dummy to make it openable
                    done=1
                except:
                    None
            if done==0 and  hasattr(variable,'__iter__'):
                try:
                    if isinstance(variable, np.ndarray):
                        values='numpy.ndarray\xa0'+str(variable.shape)
                        self.trvw.insert(parent_entry, 'end',parent_entry+'%'+entry, text=entry,values=values.replace(' ','\xa0'),tags='color2')
                    else:
                        values=str(type(variable))
                        if len(values)>25:
                            values=values[0:10]+'..'
                        self.trvw.insert(parent_entry, 'end',parent_entry+'%'+entry, text=entry,values=values.replace(' ','\xa0'),tags='color2')
                    self.trvw.insert(parent_entry+'%'+entry, 'end', text=' ') #dummy to make it openable
                    done=1
                except:
                    None
            if done==0:
                try:
                    values=str(variable)
                    if len(values)>25:
                        values=values[0:25]+'..'
                    self.trvw.insert(parent_entry, 'end',parent_entry+'%'+entry, text=entry,values=values.replace(' ','\xa0'))
                except ValueError:
                    values=str(type(variable))
                    if len(values)>25:
                        values=values[0:25]+'..'
                    self.trvw.insert(parent_entry, 'end',parent_entry+'%'+entry, text=entry,values=values.replace(' ','\xa0'))
            self.entrdct[parent_entry+'%'+entry]=variable
            self.strdict[parent_entry+'%'+entry]=self.strdict[parent_entry]+string

    def branch_open(self, event=None,value=None):
        entries = []
        variables = []
        strings=[]
        dict_or_dir=self.dict_or_dir.get()
        max_children=int(self.max_children_string.get())
        children=0
        if value==None:
            value=self.trvw.focus()
        for child in self.trvw.get_children(value):
            self.trvw.delete(child)
        if isinstance(self.entrdct[value], dict):
            try:
                for i,key in enumerate(self.entrdct[value].keys()):
                    if children>max_children-1:
                        break
                    entries.append('['+key+']')
                    variables.append(self.entrdct[value][key])
                    if isinstance(key, str):
                        strings.append('["'+key+'"]')
                    elif isinstance(key, int):
                        strings.append('['+str(key)+']')
                    else:
                        strings.append('[list('+self.strdict[value]+'keys()'+')['+str(i)+']]')
                    children+=1
            except:
                None
        elif hasattr(self.entrdct[value],'keys') and hasattr(self.entrdct[value].keys,'__call__'):#str(type(self.entrdct[value]))=="<class 'h5py._hl.files.File'>" or str(type(self.entrdct[value]))=="<class 'h5py._hl.group.Group'>" :
            try:
                for key in self.entrdct[value].keys(): # get files in selected forlder
                    if children>max_children-1:
                        break
                    entries.append('['+key+']')
                    variables.append(self.entrdct[value][key])
                    strings.append('["'+key+'"]')
                    children+=1
            except:
                None
        if dict_or_dir==0 and hasattr(self.entrdct[value],'attrs') and 'attrs' not in self.entrdct[value].__dict__:
            try:
                for key in self.entrdct[value].attrs.keys():
                    if children>max_children-1:
                        break
                    entries.append('.attrs['+str(key)+']')
                    variables.append(self.entrdct[value].attrs[key])
                    strings.append('.attrs['+key+']')
                    children+=1
            except:
                None
        if dict_or_dir==0 and hasattr(self.entrdct[value],'__dict__'):
            try:
                for key in self.entrdct[value].__dict__.keys():
                    if children>max_children-1:
                        break
                    entries.append('.'+key)
                    variables.append(self.entrdct[value].__dict__[key])
                    strings.append('.'+key)
                    children+=1
            except:
                None
        if dict_or_dir==1 and hasattr(self.entrdct[value],'__dir__'):
            try:
                for key in dir(self.entrdct[value]):
                    if children>max_children-1:
                        break
                    entries.append('.'+key)
                    variables.append(getattr(self.entrdct[value],key))
                    strings.append('.'+key)
                    children+=1
            except:
                None
        elif hasattr(self.entrdct[value],'__iter__'):
            try:
                for i,_ in enumerate(self.entrdct[value]):
                    if children>max_children-1:
                        break
                    variables.append(self.entrdct[value][i])
                    entries.append(i)
                    strings.append('['+str(i)+']')
                    children+=1
            except:
                None
        if len(variables)>0:
            self.add(value,entries,variables,strings)
        self.get_widt()

    def branch_close(self, event=None):
        for child in self.trvw.get_children(self.trvw.focus()):
            self.trvw.delete(child)
        self.trvw.insert(self.trvw.focus(), 'end', text=' ') #dummy to make it openable
        self.get_widt()

    def get_widt(self):
        entries=[]
        for child in self.trvw.get_children():
            entries.append([0,child])
        i=0
        max_length=230
        max_length2=100
        while i<len(entries):
            for child in self.trvw.get_children(entries[i][1]):
                entries.append([entries[i][0]+1,child])
            text=str(self.trvw.item(entries[i][1])['text'])
            #length=20+entries[i][0]*20+tkinter.font.Font().measure(text)*0.85# Too slow in practice
            length=25+entries[i][0]*20+len(text)*self.monospace_font_widt
            #print(self.trvw.bbox(entries[i][1],column='#0'))
            if length>max_length:
                max_length=length
            if len(self.trvw.item(entries[i][1])['values'])>0:
                text=str(self.trvw.item(entries[i][1])['values'][0])
                #length=20+entries[i][0]*20+tkinter.font.Font().measure(text)*0.85# Too slow in practice
                length=len(text)*self.monospace_font_widt
                if length>max_length2:
                    max_length2=length
            i+=1
        self.trvw.column("#0", width=int(max_length2)+int(max_length)-190)
        self.trvw['columns'] = ()
        self.trvw['columns'] = [1]
        self.trvw.column("#0", width=int(max_length))
        self.trvw.column("#1", width=int(max_length2)+10)
        #self.trvw['columns'] = [1]

    def rebuild(self):
        #get a set of open folders
        set_of_open_entries=set()
        entries=[]
        for child in self.trvw.get_children():
            entries.append(child)
        i=0
        while i<len(entries):
            for child in self.trvw.get_children(entries[i]):
                entries.append(child)
            if self.trvw.item(entries[i], 'open'):
                set_of_open_entries.add(entries[i])
            i+=1
        for child in self.trvw.get_children():
            self.trvw.delete(child)
        entries = []
        variables = []
        strings = []
        if self.simple_space_var.get():
            for key in self.parent.__dict__.keys():
                entries.append(key)
                variables.append(self.parent.__dict__[key])
                strings.append('self.'+key)
        else:
            if hasattr(self.parent,'run'):
                for key in self.parent.run.__dict__:
                    entries.append(key)
                    variables.append(self.parent.run.__dict__[key])
                    strings.append('self.run.'+key)
        if len(entries)>0:
            self.add('',entries,variables,strings)
        entries=[]
        #open previously opened files
        for child in self.trvw.get_children():
            entries.append(child)
        i=0
        while i<len(entries):
            if entries[i] in set_of_open_entries:
                self.branch_open(self,value=entries[i])
                self.trvw.item(entries[i],open=True)
            for child in self.trvw.get_children(entries[i]):
                try:
                    entries.append(child)
                except:
                    None
            i+=1
        return
