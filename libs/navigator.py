import tkinter #from Tkinter import StringVar, Tk, Label, RIGHT, BOTH, RAISED, Text, TOP, BOTH, X, LEFT, W, N, E, S, Listbox, END,Radiobutton, IntVar, Checkbutton, DISABLED, NORMAL
from tkinter import ttk #from ttk import Frame, Button, Style, Label, Entry
import os
import time
class navigator():
    def __init__(self, parent, navigatorFrame,settings):
        '''
        Class for the navigator
        includes the listbox(lb) and (sort and rebuild buttons) as UI objects
        the listbox(lb) does not show the full path of the item, but only the filename, indented one time for each folder it is deep.
        lbpaths is a list that has the same elements as lb, but contains the full paths.
        '''
        #print(' 3a')
        self.currently_opening_closing_folder=True
        self.parent=parent
        ####################### place UI items
        self.navigatorFrame=navigatorFrame
        #style = ttk.Style()
        #style.configure("mystyle.Treeview",font='TkFixedFont',size=9)
        #print(' 3a')
        #self.parent.update() #temp
        #print(' 3a')

        navigator_tree_frame=ttk.Frame(navigatorFrame, relief=tkinter.RAISED, borderwidth=0)
        navigator_tree_frame.pack(fill=tkinter.BOTH, expand=True,side=tkinter.TOP, anchor=tkinter.E)
        #print(' 3a')
        self.trvw=ttk.Treeview(navigator_tree_frame,selectmode='extended',show="tree",style="mystyle.Treeview") # the navigator listbox
        #print(' 3b')
        self.monospace_font_widt=tkinter.font.Font(font='TkFixedFont', size=9).measure(' ')
        #print(self.monospace_font_widt)
        self.trvw.column('#0', stretch=0)
        self.trvw.bind("<<TreeviewOpen>>", self.branch_open, "+")
        self.trvw.bind("<<TreeviewClose>>", self.branch_close, "+")
        self.trvw.bind("<<TreeviewSelect>>", self.trvw_select) # bind selecting elements to open close folder # passes the event to the function
        #self.parent.update() #temp
        self.trvw.tag_configure('color0', background='#FFFFFF')
        self.trvw.tag_configure('color1', background='#7777FF')
        self.trvw.tag_configure('color2', background='#77FF77')
        self.trvw.tag_configure('color3', background='#FF7777')
        self.trvw.tag_configure('color23', background='#FFFF77')
        self.trvw.tag_configure('color13', background='#FF77FF')
        self.trvw.tag_configure('color12', background='#77FFFF')
        self.trvw.tag_configure('color123', background='#AAAAAA')
        #print(' 3b2')
        self.list_of_colored_entries={}
        vsb = ttk.Scrollbar(navigator_tree_frame, orient="vertical", command=self.trvw.yview)
        self.trvw.configure(yscrollcommand=vsb.set)
        self.trvw.pack(side=tkinter.LEFT,anchor=tkinter.S,fill=tkinter.BOTH, expand=1)
        #self.trvw.update()
        vsb.pack(side=tkinter.LEFT,anchor=tkinter.S,fill=tkinter.BOTH, expand=0)
        #print(' 3b2')
        #vsb.update()
        #print(' 3b2')
        #self.trvw.update()
        #print(' 3b2')
        #self.parent.update() #temp
        #print(' 3b3')
        '''self.lb=tkinter.Listbox(navigatorFrame,selectmode='extended',exportselection=False,width=0) # the navigator listbox
        self.lb.pack(side=tkinter.TOP,anchor=tkinter.S,fill=tkinter.BOTH, expand=1)
        # sort and rebuild buttons'''
        self.empty_frames=[]
        #print(' 3c')
        for j in range(10):
           self.empty_frames.append(
               ttk.Frame(navigatorFrame, relief=tkinter.RAISED, borderwidth=0))
           #self.empty_frames[-1].pack(side=tkinter.TOP,anchor=tkinter.E)
        self.sort_select = tkinter.IntVar()
        self.sort_select.set(settings['list_sort_type'])  # initializing the choice, i.e. 0
        rebuildframe= ttk.Frame(navigatorFrame, relief=tkinter.RAISED, borderwidth=0)
        rebuildframe.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH)
        sort_choices = [
          ("Date",0),
          ("Name",1)
        ]
        self.sort_buttons=[]
        for txt, val in sort_choices:
            self.sort_buttons.append(tkinter.Radiobutton(rebuildframe,
            text=txt,
            indicatoron=0,
            variable=self.sort_select,
            value=val,
            background="#FFFFFF"))
        for i in range(len(self.sort_buttons)):
            self.sort_buttons[i].pack(side=tkinter.LEFT, fill=tkinter.BOTH)
        dir_sort_choices = [
          ("↑",0),
          ("↓",1)
        ]
        self.sort_direction_select = tkinter.IntVar()
        self.sort_direction_select.set(settings['list_sort_direction'])  # initializing the choice, i.e. 1
        self.sort_direction_buttons=[]
        #print(' 3d')

        #self.parent.update() #temp
        for txt, val in dir_sort_choices:
            self.sort_direction_buttons.append(tkinter.Radiobutton(rebuildframe,
            text=txt,
            indicatoron=0,
            variable=self.sort_direction_select,
            value=val,
            background="#FFFFFF"))
        for i in range(len(self.sort_direction_buttons)):
            self.sort_direction_buttons[i].pack(side=tkinter.LEFT, fill=tkinter.BOTH)
        self.rebuildbutton=ttk.Button(rebuildframe, text="Rebuild", command=self.rebuild)
        self.rebuildbutton.pack(side=tkinter.LEFT,fill=tkinter.BOTH)
        self.clear_button=ttk.Button(rebuildframe, text="Clear selection", command=self.deselect)
        self.clear_button.pack(side=tkinter.LEFT,fill=tkinter.BOTH)
        self.root_folder=parent.settings['root_folder']
        if str(self.root_folder) == '%Path_of_program%/':
            self.root_folder=''
        #print(' 3d')
        #self.parent.update() #temp
        #parent.after(20, self.pupulate_initial_folder)


    def pupulate_initial_folder(self):
        #self.parent.update()
        self.trvw.insert('', 'end', self.root_folder+'../',text='../',values=self.root_folder+'../')
        self.trvw.insert(self.root_folder+'../', 'end', text=' ') #dummy to make it openable
        files = []
        if not self.root_folder=='':
            for f in os.listdir(self.root_folder): # get files in selected forlder
                if not f[0]=='.':
                    files.append(f)
        else:
            for f in os.listdir(): # get files in selected forlder
                if not f[0]=='.':
                    files.append(f)
        files=self.sort_files(self.root_folder,files)
        self.add(self.root_folder,files,is_root=True)
        self.currently_opening_closing_folder=False
        #self.parent.update()
        '''
        # end sort and rebuild buttons
        ####################### Initiate listbox
        self.lb.insert(tkinter.END, '../') # add 'upp one folder to lb
        self.lb.itemconfig(0,{'bg':'grey90'}) # set it to be grey, to mark it as a folder and not a file
        self.isopen=dict() # dict that describes if a folder is open or not
        self.lbpaths=['.']# set first element to current folder, so this can be easily opened by calling self.folderopenclose(0)
        self.folderopenclose(0) # open the current folder
        self.lbpaths[0]='../'  # correct first element to be 'one folder upp'
        self.lb.bind("<<ListboxSelect>>", self.onselect) # bind selecting elements to open close folder # passes the event to the function'''
    def branch_open(self, event=None,entry=None):
        if self.currently_opening_closing_folder==True:
            return
        self.currently_opening_closing_folder=True
        if entry==None:
            entry=self.trvw.focus()
        if not os.path.isdir(entry):
            self.currently_opening_closing_folder=False
            return
        for child in self.trvw.get_children(entry):
            self.trvw.delete(child)
        item=self.trvw.item(entry)
        value=entry
        files = []
        for f in os.listdir(value): # get files in selected forlder
            if not f[0]=='.':
                files.append(f)
        sortpar=[]
        files=self.sort_files(value+'/', files)
        self.add(value,files)
        self.get_widt()
        self.currently_opening_closing_folder=False
    def sort_files(self, folder, files):
        sortpar=[]
        if self.sort_select.get()==0: # sort files on date modified
            for f in files:
                sortpar.append(os.path.getmtime(folder+f))
        elif self.sort_select.get()==1: # sort files on name
            for f in files:
                sortpar.append(f.lower())
        if self.sort_direction_select.get()==0:
            files=[f for (t,f) in sorted(zip(sortpar,files), reverse=True)]
        else:
            files=[f for (t,f) in sorted(zip(sortpar,files), reverse=False)]
        return files

    def add(self,value,files,is_root=False):
        if is_root==False:
            parent_value=value
        else:
            parent_value=''
        if value.replace('../','')==self.root_folder and not value==self.root_folder:
            self.trvw.insert(value, 'end',value+'../', text='../',values=value+'../')
            self.trvw.insert(value+'../', 'end', text=' ') #dummy to make it openable
            #self.parent.update()
            time.sleep(0.001)
        for file in files:
            if os.path.isfile(value+file):
                if file.split('.')[-1] in self.parent.recognised_files:
                    self.trvw.insert(parent_value, 'end',value+file, text=file)
                    if value+file in self.list_of_colored_entries:
                        tag=self.list_of_colored_entries[value+file][1]
                        self.trvw.item(value+file,tags=tag)
                        self.list_of_colored_entries[value+file]=(value+file,tag)
                    ##self.parent.update()
                    #time.sleep(0.001)
            else:
                self.trvw.insert(parent_value, 'end',value+file+'/', text=file)
                if os.path.isdir(value+file+'/'):
                    self.trvw.insert(value+file+'/', 'end', text=' ') #dummy to make it openable
                #self.parent.update()
                time.sleep(0.001)
    def branch_close(self, event=None):
        if self.currently_opening_closing_folder==True:
            return
        self.currently_opening_closing_folder=True
        for child in self.trvw.get_children(self.trvw.focus()):
            self.trvw.delete(child)
        value=self.trvw.focus()
        self.trvw.insert(value, 'end', text=' ') #dummy to make it openable
        self.get_widt()
        self.currently_opening_closing_folder=False
    def trvw_select(self,event=None):
        selected_items=self.trvw.selection()
        if len(selected_items)==1:
            value=selected_items[0]
            self.parent.add_buttons(value) # add buttons corresponding to filetype
    def get_widt(self):
        entries=[]
        for child in self.trvw.get_children():
            entries.append([0,child])
        i=0
        max_length=230
        while i<len(entries):
            for child in self.trvw.get_children(entries[i][1]):
                entries.append([entries[i][0]+1,child])
            text=self.trvw.item(entries[i][1])['text']
            #length=20+entries[i][0]*20+tkinter.font.Font().measure(text)*0.85# Too slow in practice
            length=25+entries[i][0]*20+len(text)*self.monospace_font_widt
            #print(self.trvw.bbox(entries[i][1],column='#0'))
            if length>max_length:
                max_length=length
            i+=1
        #self.navigatorFrame.grid_propagate(False)
        #self.navigatorFrame.config(width=int(max_length)//10)
        self.trvw.column("#0", width=int(max_length))
        self.trvw['columns'] = ()
        #self.trvw.config( width=int(max_length) )
        #self.navigatorFrame.update()

    def color_selected(self,tag):
        selected_items=self.trvw.selection()
        for handle in selected_items:
            value=handle
            item=self.trvw.item(handle)
            tags=item['tags']
            set_of_tags=split_color_tags(tags)
            if not tag in set_of_tags:
                set_of_tags.add(tag)
            new_tag=merge_color_tags(set_of_tags)
            self.trvw.item(handle,tags=new_tag)
            self.list_of_colored_entries[value]=(handle,new_tag)
        return
    def clear_color(self,tag):
        for key, value in  self.list_of_colored_entries.items():
            try:
                item=self.trvw.item(value[0])
                tags=item['tags']
                if not tags=='':
                    set_of_tags=split_color_tags(tags)
                    set_of_tags.remove(tag)
                    new_tag=merge_color_tags(set_of_tags)
                    self.trvw.item(value[0],tags=new_tag)
                    if new_tag=='':
                        self.list_of_colored_entries.remove(key)
                    else:
                        self.list_of_colored_entries[key]=(value[0],new_tag)
            except:
                None
        return
    def rebuild(self):
        #get a set of open folders
        set_of_open_folders=set()
        entries=[]
        for child in self.trvw.get_children():
            entries.append(child)
        i=0
        while i<len(entries):
            for child in self.trvw.get_children(entries[i]):
                entries.append(child)
            if self.trvw.item(entries[i], 'open'):
                value=entries[i]
                set_of_open_folders.add(value)
            i+=1
        for child in self.trvw.get_children():
            self.trvw.delete(child)
        #repopulate list with current folder
        self.trvw.insert('', 'end', '../',text='../',values='../')
        self.trvw.insert('../', 'end', text=' ') #dummy to make it openable
        files = []
        for f in os.listdir(): # get files in selected forlder
            if not f[0]=='.':
                files.append(f)
        files=self.sort_files('',files)
        self.add('',files)
        entries=[]
        #open previously opened files
        for child in self.trvw.get_children():
            entries.append(child)
        i=0
        while i<len(entries):
            value=entries[i]
            if value in set_of_open_folders:
                self.branch_open(self,entry=entries[i])
                self.trvw.item(entries[i],open=True)
            for child in self.trvw.get_children(entries[i]):
                try:
                    entries.append(child)
                except:
                    None
            i+=1
        return
    def get_paths_of_selected_items(self):
        selected_items=self.trvw.selection()
        files=[]
        for i,_ in enumerate(selected_items):
            value=selected_items[i]
            if os.path.exists(value):
                files.append(value)
        return files

    def deselect(self):
        while len(self.trvw.selection()) > 0:
            self.trvw.selection_remove(self.trvw.selection()[0])
# custom maplotlib toolbar. Removes 'Back' and 'Forward' buttons
import copy
single_to_multiple_color_dict={
    '':set([]),
    'color1':set(['color1']),
    'color2':set(['color2']),
    'color3':set(['color3']),
    'color12':set(['color1','color2']),
    'color23':set(['color2','color3']),
    'color13':set(['color1','color3']),
    'color123':set(['color1','color2','color3']),
}
def split_color_tags(tags):
    global single_to_multiple_color_dict
    if len(tags)==0:
        tags=['']
    return(copy.deepcopy(single_to_multiple_color_dict[str(tags[0])]))

multiple_to_single_color_dict={
    '':'',
    'color1':'color1',
    'color2':'color2',
    'color3':'color3',
    'color1 color2':'color12',
    'color2 color3':'color23',
    'color1 color3':'color13',
    'color1 color2 color3':'color123',
}
def merge_color_tags(set_of_tags):
    list_of_tags=sorted(set_of_tags)
    global multiple_to_single_color_dict
    return([multiple_to_single_color_dict[' '.join(list_of_tags)]])
