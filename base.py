    #!/usr/bin/python
# -*- coding: utf-8 -*-


from libs.navigator import navigator
from libs.data_tree import data_tree
from libs.turn_onoff_modules import turn_onoff_modules
import fns
import traceback

import gc
class mainFrame(ttk.Frame):  # main window
    def __init__(self, parent, settings):
        #print(' 1')
        self.settings=settings
        self.parent = parent
        self.main_thread=threading.currentThread()
        # make main frame (self now 'becomes' the frame)
        ttk.Frame.__init__(self, parent)
        ##self.update() #temp
        # set window title, icon
        self.parent.title("Data manager")
        _, ICON_PATH = tempfile.mkstemp()
        with open(ICON_PATH, 'wb') as icon_file:
            icon_file.write(ICON)  # icon loaded in TDM.py
        self.parent.wm_iconbitmap(bitmap='@'+ICON_PATH)

        # configure the main frame, that all other frames are placed on

        self.pack(fill=tkinter.BOTH, expand=True)
        self.style = ttk.Style()
        self.style.element_create('Plain.Notebook.tab', "from", 'default')
        # the theme used use #print(self.style.theme_names())# to see list off possible themes
        self.style.theme_use("clam")
        # white background in buttons with unspecified style
        self.style.configure('TButton', foreground="black", background="white")
        self.style.configure('color0.TButton')
        self.style.configure('color1.TButton', background="#7777FF")
        self.style.map('color1.TButton',
        background=[('disabled', '#7777FF'),('pressed', '!focus', '#AAAAFF'),('active', '#5555DD')],)
        self.style.configure('color2.TButton', background="#77FF77")
        self.style.map('color2.TButton',
        background=[('disabled', '#77FF77'),('pressed', '!focus', '#AAFFAA'),('active', '#55DD55')],)
        self.style.configure('color3.TButton', background="#FF7777")
        self.style.map('color3.TButton',
        background=[('disabled', '#FF7777'),('pressed', '!focus', '#FFAAAAA'),('active', '#DD5555')],)
        self.style.configure('color23.TButton', background="#FFFF77")
        self.style.map('color23.TButton',
        background=[('disabled', '#FFFF77'),('pressed', '!focus', '#FFFFAA'),('active', '#DDDD55')],)
        self.style.configure('color13.TButton', background="#FF77FF")
        self.style.map('color13.TButton',
        background=[('disabled', '#FF77FF'),('pressed', '!focus', '#FFAAFF'),('active', '#DD55DD')],)
        self.style.configure('color12.TButton', background="#77FFFF")
        self.style.map('color12.TButton',
        background=[('disabled', '#77FFFF'),('pressed', '!focus', '#AAFFFF'),('active', '#55DDDD')],)
        self.style.configure('color123.TButton', background="#AAAAAA")
        self.style.map('color123.TButton',
        background=[('disabled', '#AAAAAA'),('pressed', '!focus', '#BBBBBB'),('active', '#888888')],)
        # white background in frames with unspecified style
        self.style.configure("TFrame", background="white")
        self.style.configure("TLabel", background="white")
        # white background in Checkbuttons with unspecified style
        self.style.configure("TCheckbutton", background="white")
        self.style.configure("TNotebook", background="white")
        self.style.map("TNotebook.Tab",background=[('selected', 'White'),('active', '#EEEEEE')])
        self.style.configure("TNotebook.Tab", background='#DDDDDD');

        self.style.configure("TNotebook", background="white")
        #print(' 2')
        #    0     1
        # #----#--------#
        # |     |file run|
        # |     |options | 0
        # | - - #--------#
        # |     |        |
        # |trvw |  fig   | 1
        # |-----|        |
        # |outp |        | 2
        # |     #--------#
        # |     |tool,cmd| 3
        # #-----#--------#
        #

        # weight=1-> expandable, i.e. row 1, col 1 will expand to fill the frame # see drawing above for map of rows and columns
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        # Map of the main window
        # Nav is the navigator, out is the output
        #
        # #-----#-------------------------#
        # |#---#| | top_frame             |
        # ||   || | #-----filename------# |
        # ||   || | |    --------->     | |
        # ||   || | #-------------------# |
        # ||   || | #---buttonframe1----# |
        # ||   || | |    --------->     | |
        # ||   || | #-------------------# |
        # ||Nav|| |         ...           |
        # ||   || | #---buttonframe10---# |
        # ||   || | |    --------->     | |
        # ||   || v #-------------------# |
        # ||   |#-------------------------#
        # ||   ||                         |
        # ||   ||                         |
        # |#---#|          figure         |
        # |#---#|                         |
        # ||   ||                         |
        # ||out|#-------------------------#
        # ||   ||  toolbar_frame          |
        # |#---#|                         |
        # #-----#-------------------------#
        #
        # data_tree
        data_tree_frame_parent= ttk.Frame(self, relief=tkinter.RAISED, borderwidth=0)
        data_tree_frame_parent.grid(row=0, column=2, columnspan=1, rowspan=4,
                             padx=5, sticky=tkinter.E+tkinter.W+tkinter.S+tkinter.N)
        show_data_tree_button = ttk.Button(data_tree_frame_parent, text="d\ne\nv\n \nt\no\no\nl\ns", width =1, command=self.show_data_tree)
        show_data_tree_button.pack(side=tkinter.LEFT,expand=True)
        self.data_tree_frame = ttk.Frame(data_tree_frame_parent, relief=tkinter.RAISED, borderwidth=0)

        #self.data_tree_frame.pack(fill=tkinter.BOTH, expand=True,side=tkinter.LEFT, anchor=tkinter.E)

        self.console_intput = tkinter.Text(self.data_tree_frame, width=0, height=5)
        self.console_intput.pack(side=tkinter.BOTTOM,fill=tkinter.BOTH, expand=0)
        self.console_intput.bind("<Shift-Return>", self.execute_command)
        self.console_intput.bind('<Shift-Up>', self.set_pervious_command)
        self.console_intput.bind('<Shift-Down>', self.set_next_command)
        self.console_output_box = tkinter.Text(self.data_tree_frame, width=0, height=20)
        self.console_output_box.pack(side=tkinter.BOTTOM,fill=tkinter.BOTH, expand=0)
        self.module_enable_frame = ttk.Frame(self.data_tree_frame, relief=tkinter.RAISED, borderwidth=0)
        self.module_enable_frame.pack(side=tkinter.TOP,fill=tkinter.BOTH, expand=0)
        self.module_reload_var = tkinter.IntVar()
        self.module_reload_var.set(settings['reload_check'])
        self.module_reload = ttk.Checkbutton(
            self.module_enable_frame, text="reload modules on run", variable=self.module_reload_var)
        self.module_reload.pack(side=tkinter.LEFT)
        self.reload_modules_button = ttk.Button(
            self.module_enable_frame, text="reload modules", command=self.import_modules)
        self.reload_modules_button.pack(side=tkinter.LEFT)
        self.modules_to_run = tkinter.Label(
            self.module_enable_frame, text='| Enabled modules:', background="#FFFFFF")
        self.modules_to_run.pack(side=tkinter.LEFT)
        self.onoff_module_buttons=turn_onoff_modules(self.data_tree_frame)
        #self.update()
        self.data_tree = data_tree(self, self.data_tree_frame)
        #self.update()
        #print(' 3')

        # navigator
        # set recognised file extensions and load modules:
        num_recognized_classes = self.import_modules(force=False)
        navigator_frame = ttk.Frame(self, relief=tkinter.RAISED, borderwidth=0)
        navigator_frame.grid(row=0, column=0, columnspan=1, rowspan=2,
                             padx=5, sticky=tkinter.E+tkinter.W+tkinter.S+tkinter.N)
        self.nav = navigator(self, navigator_frame,settings)
        # output_box
        self.output_box = tkinter.Text(self, width=0, height=15)
        self.output_box.grid(row=2, column=0, columnspan=1, rowspan=2,
                             padx=5, sticky=tkinter.E+tkinter.W+tkinter.S+tkinter.N)

        # output_box
        # figure
        #print(' 4')
        self.figure = matplotlib.figure.Figure(dpi=80, facecolor='w', edgecolor='k')
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(
            self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=1, column=1, rowspan=2, columnspan=1,
                                         sticky=tkinter.E+tkinter.W+tkinter.S+tkinter.N)
        self.canvas.mpl_connect('button_press_event', self.callbackClick)
        #set up a hidden figure for plotting with custom SG_size
        self.hidden_figure = matplotlib.figure.Figure(dpi=80, facecolor='w', edgecolor='k')
        self.hiddne_canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(
        self.hidden_figure, master=self)
        # figure done
        # top_frame
        #print(' 4a')

        ##self.update() #temp
        top_frame = ttk.Frame(self, relief=tkinter.RAISED, borderwidth=0)
        top_frame.grid(row=0, column=1, columnspan=1, sticky=tkinter.E+tkinter.W)
        # filename and runbutton etc.
        filename_frame = ttk.Frame(
            top_frame, relief=tkinter.RAISED, borderwidth=0)
        filename_frame.pack(side=tkinter.TOP, anchor=tkinter.E)
        date_today=str(datetime.date.today())
        self.default_filename=settings['save_path'].replace('%date%',date_today)
        default_folder='/'.join(self.default_filename.split('/')[0:-1])
        if not (os.path.exists(default_folder) and not os.path.isfile(default_folder)):
            os.makedirs(default_folder)
        self.name_field_string = tkinter.StringVar()
        self.name_field = ttk.Entry(
            filename_frame, width=40, textvariable=self.name_field_string)
        self.name_field.pack(side=tkinter.LEFT)
        self.save_check_var = tkinter.IntVar()
        self.save_check_var.set(settings['save_check'])
        self.save_ckeck = ttk.Checkbutton(
            filename_frame, text="Save", variable=self.save_check_var)
        self.save_ckeck.pack(side=tkinter.LEFT)
        run = ttk.Button(filename_frame, text="Run",
                         command=self.runbuttonpress)
        run.pack(side=tkinter.LEFT)
        self.threading_check_var = tkinter.IntVar()
        self.threading_check = ttk.Checkbutton(
            filename_frame, text="Threading", variable=self.threading_check_var, command=self.toggle_threading)
        self.threading_check.pack(side=tkinter.LEFT)
        self.thread_label = tkinter.Label(
            filename_frame, text='Threads: 0', background="#FFFFFF")
        self.stop = ttk.Button(filename_frame, text="Stop",
                               command=self.stopbuttonpress)
        if settings['threading']:
            self.threading_check_var.set(1)
            self.toggle_threading()
        # set default plot_number for saved plots
        self.plot_number = 1
        while os.path.isfile(self.default_filename+str(self.plot_number)+'.png') or os.path.exists(self.default_filename+str(self.plot_number)):
            self.plot_number += 1
        self.name_field_string.set(self.default_filename+str(self.plot_number))
        # filename done
        # dynamically allocated button frames:
        self.notebook = ttk.Notebook(top_frame, name='nb')
        self.notebook.pack(side=tkinter.TOP, anchor=tkinter.E,fill=tkinter.BOTH)
        self.notebook_tabs = []
        self.notebook_tab_labels = []
        self.button_frames = []  # 2d , [tab][row]

        #self.update() #temp
        #print(' 5')
        self.notebook_style = [('Plain.Notebook.tab', {'children':
                                                      [('Notebook.padding', {'side': 'top', 'children':
                                                                             [('Notebook.focus', {'side': 'top', 'children':
                                                                                                  [('Notebook.label', {
                                                                                                    'side': 'top', 'sticky': ''})],
                                                                                                  'sticky': 'nswe'})],
                                                                             'sticky': 'nswe'})],
                                                      'sticky': 'nswe'})]
        try:
            self.style.layout('TNotebook.Tab', self.notebook_style)
        except Exception as e:
            print(e)
        #print(' 6')
        #self.update() #temp
        for i in range(10):
            self.notebook_tabs.append(tkinter.Frame(
                self.notebook, name=''+str(i),bg='white'))
            self.notebook_tab_labels.append('Tab '+str(i))
            # self.notebook.add(self.notebook_tabs[-1], text='tab '+str(i))
            # self.notebook.forget(self.notebook_tabs[3])
            self.button_frames.append([])  # 2d , [tab][row]
            for j in range(10):
                self.button_frames[-1].append(
                    ttk.Frame(self.notebook_tabs[-1], relief=tkinter.RAISED, borderwidth=0))
                self.button_frames[-1][-1].pack(side=tkinter.TOP,
                                               anchor=tkinter.E,fill=tkinter.BOTH)
        self.button_frames.append(self.nav.empty_frames)  # 'tab11' is below the file browser
        self.buttons = {}  # stores dynamically allocated buttons
        self.button_handles = {}  # stores handles for dynamically allocated buttons
        #print(' 7')
        # self.curclasses=[]
        # toolbar_frame
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.grid(row=3, column=1,
                               rowspan=1, columnspan=1)
        self.canvastoolbar = CustomToolbar(self.canvas, self.toolbar_frame)
        self.canvastoolbar.config(background="white")
        self.canvastoolbar._message_label.config(background="white")
        self.canvastoolbar.pack(side=tkinter.LEFT)
        #self.canvastoolbar.update()

        # declare additional variables related to threading, and multiprocessing
        self.quit_flag = 0
        self.stop_flag = 0
        # The pool class needs to have a pointer to this class, so that it can check stop_flag
        fns.Pool.frame = self
        self.pools = []
        self.matplotlib_plot_num = 1000
        #print(' 8')

        self.purge_buttons()
        #print(' 9')

        self.current_filetype=''
        if num_recognized_classes==1:
            key=self.recognised_files.__iter__().__next__() #gets first key in dict
            self.add_buttons('.'+key)
        # redirect stdout to the output_box
        #print(' 10')
        sys.stdout = Std_redirector(self.output_box)
        fns.decorate_matplotlib()
        sys.stderr = Std_redirector(self.console_output_box)
        self.nav.pupulate_initial_folder()
        self.old_threads=0
        self.after(1000,self.count_thread)
        #print(' 11')

    # add dynamically allocated buttons when a recognized file is selected:
    def import_modules(self,force=True):
        # import all modules in pkg_dir
        pkg_dir = 'modules'
        module_classes=[]
        for (module_loader, name, ispkg) in pkgutil.iter_modules([pkg_dir]):
            if 'modules.'+name in sys.modules:
                try:
                    if self.module_reload_var.get() or force==True:
                        importlib.reload(sys.modules['modules.'+name])
                    if hasattr(sys.modules['modules.'+name],'moduleClass'):
                        if self.onoff_module_buttons.make_or_get_modulebutton(name):
                            module_classes.append(sys.modules['modules.'+name].moduleClass)
                except Exception as e:
                    traceback.print_exc()
                    eprint(e)
                    try:
                        errortext=traceback.format_exc()
                        exceptiont_to_fig(self,errortext)
                    except:
                        None
            else:
                try:
                    importlib.import_module('modules.'+name)
                    if hasattr(sys.modules['modules.'+name],'moduleClass'):
                        if self.onoff_module_buttons.make_or_get_modulebutton(name):
                            module_classes.append(sys.modules['modules.'+name].moduleClass)
                except Exception as e:
                    traceback.print_exc()
                    eprint(e)
                    try:
                        errortext=traceback.format_exc()
                        exceptiont_to_fig(self,errortext)
                    except:
                        None
        # set recognised file types
        recognised_files = {}
        for c in module_classes:
            for ending in c.filetypes:
                if ending not in recognised_files:
                    recognised_files[ending] = [c]
                else:
                    recognised_files[ending].append(c)
        self.recognised_files=recognised_files
        return len(module_classes)

    def add_buttons(self, path):
        ending = path.split('.')[-1]
        if ending==self.current_filetype:
            return
        # adds buttons to self.button_frames[i]
        # a handle is added to self.button_handles
        # the modlues are responsible for providing unique handles
        self.current_filetype=ending
        if not '.' in path.split('/')[-1] or ending=='':
            return
        # abort if already adding buttons
        if hasattr(self,'currently_adding_buttons'):
            if self.currently_adding_buttons==True:
                return
        self.currently_adding_buttons=True
        prev_buttons = list(self.buttons.keys())
        # for all modules that recognize this filetype
        for c in self.recognised_files[ending]:
            for b in c.addButtons():  # for all buttons to be added by current module
                if 1: #try:
                    if not b['key'] in prev_buttons:  # if button is not allready added -> add it
                        type=b['type']
                        if 'radio' in type:
                            # set the int to be in the map buttons
                            self.buttons[b['key']] = tkinter.IntVar()
                            # initializing the choice, i.e. 1
                            if 'default' in b:
                                self.buttons[b['key']].set(b['default'])
                            else:
                                self.buttons[b['key']].set(0)
                            self.button_handles[b['key']] = []
                            num_radionbuttons=len(b['texts'])
                            bframe,tab=get_row_tab(b)
                            for i,button_text in enumerate(b['texts']):
                                if 'vertical' in type:
                                    current_bframe=bframe+i
                                else:
                                    current_bframe=bframe
                                self.button_handles[b['key']].append(tkinter.Radiobutton(self.button_frames[tab][current_bframe],
                                                                                    text=button_text,
                                                                                    # padx = 20,
                                                                                    # borderwidth=0,
                                                                                    indicatoron=0,
                                                                                    variable=self.buttons[b['key']],
                                                                                    # command=self.Selectcolor,
                                                                                    value=i,
                                                                                    background="#FFFFFF"))
                                self.button_handles[b['key']
                                                   ][-1].pack(side=tkinter.LEFT)
                            if 'text' in type:
                                self.buttons[b['key']].button_return_type='radio:text'
                                self.buttons[b['key']].button_return_texts=b['texts']
                        elif type[0:3] == 'txt':
                            bframe,tab=get_row_tab(b)
                            self.buttons[b['key']] = tkinter.StringVar()
                            if 'default' in b:
                                self.buttons[b['key']].set(b['default'])
                            self.button_handles[b['key']] = []
                            w = 4
                            if 'width' in b:
                                w = b['width']
                            self.button_handles[b['key']].append(tkinter.Label(
                                self.button_frames[tab][bframe], text=b['text'], background="#FFFFFF"))
                            self.button_handles[b['key']][-1].pack(side=tkinter.LEFT)
                            self.button_handles[b['key']].append(
                                ttk.Entry(self.button_frames[tab][bframe], width=w, textvariable=self.buttons[b['key']]))
                            self.button_handles[b['key']][-1].pack(side=tkinter.LEFT)
                            self.buttons[b['key']].button_return_type=type
                        elif type == 'check':
                            bframe,tab=get_row_tab(b)
                            self.buttons[b['key']] = tkinter.IntVar()
                            if 'default' in b:
                                self.buttons[b['key']].set(b['default'])
                            else:
                                self.buttons[b['key']].set(0)
                            self.button_handles[b['key']] = []
                            self.button_handles[b['key']].append(ttk.Checkbutton(
                                self.button_frames[tab][bframe], text=b['text'], variable=self.buttons[b['key']]))
                            self.button_handles[b['key']][-1].pack(side=tkinter.LEFT)
                        elif type == 'label':
                            bframe,tab=get_row_tab(b)
                            # need to be included in order to remve the button correctly
                            self.buttons[b['key']] = tkinter.IntVar()
                            # self.buttons[b['key']].set(0)
                            self.button_handles[b['key']] = []
                            self.button_handles[b['key']].append(
                                ttk.Label(self.button_frames[tab][bframe], text=b['text']))
                            self.button_handles[b['key']][-1].pack(side=tkinter.LEFT)
                        elif type == 'click':
                            bframe,tab=get_row_tab(b)
                            self.button_handles[b['key']] = []
                            if 'color' in b:
                                self.button_handles[b['key']].append(
                                    ttk.Button(self.button_frames[tab][bframe], text=b['text'],style = b['color']+'.TButton'))
                            else:
                                self.button_handles[b['key']].append(
                                    ttk.Button(self.button_frames[tab][bframe], text=b['text']))
                            self.button_handles[b['key']][-1].pack(side=tkinter.LEFT)
                            self.buttons[b['key']] = self.button_handles[b['key']][-1]
                            if 'bind' in b:
                                self.buttons[b['key']].bind("<Button-1>", b['bind'])
                            if 'width' in b:
                                self.buttons[b['key']].config( width = b['width'] )
                        elif type == 'tabname':
                            tab = int(b['tab'])
                            self.notebook_tab_labels[tab] = b['text']
                        #self.update()
                        #time.sleep(0.01)
        self.purge_buttons(prev_buttons=prev_buttons, ending=ending)
                #except:
                #        print('error parsing button ',b)
                #        print('see examples for syntax')
        self.currently_adding_buttons=False


    def purge_buttons(self, prev_buttons=None, ending=''):
        #self.update()
        if prev_buttons==None:
            prev_buttons = list(self.buttons.keys())
        try:
            user_defined_classes=self.recognised_files[ending]
        except Exception as e:
            #print(e)
            user_defined_classes=[]
        for b in prev_buttons:  # go through all buttons, remove buttons not used by currently selected filetype
            isuseful = 0
            for c in user_defined_classes:
                for bb in c.addButtons():
                    if b == bb['key']:
                        isuseful = 1
            if isuseful == 0:
                for bb in self.button_handles[b]:
                    bb.destroy()
                self.button_handles.pop(b)
                self.buttons.pop(b)
                #self.update()
                #time.sleep(0.01)
        used_tabs=[]
        #print(' 8a')
        for i in range(10):  # go thorugh all button_frames, and show/hide them as required
            try:
                selected_tab=self.notebook.index("current")
            except Exception as e:
                #print(e)
                selected_tab=0
            tab_used = False
            for j in range(10):  # go thorugh all button_frames, and show/hide them as required
                if len(self.button_frames[i][j].winfo_children()) == 0:
                    self.button_frames[i][j].pack_forget()
                else:
                    self.button_frames[i][j].pack(
                        side=tkinter.TOP, anchor=tkinter.E,fill=tkinter.BOTH)
                    tab_used = True
            if tab_used:
                used_tabs.append(i)
            else:
                try:
                    self.notebook.forget(self.notebook_tabs[i])
                    #self.update()
                    #time.sleep(0.01)
                except Exception as e:
                    #print(e)
                    None
        #print(' 8b')
        for i in used_tabs:
            self.notebook.add(
                self.notebook_tabs[i], text=self.notebook_tab_labels[i])
            #self.update()
            #time.sleep(0.01)
        if not selected_tab in used_tabs and len(used_tabs)>0:
            self.notebook.select(used_tabs[0])
            #self.update()
            #time.sleep(0.01)
                # toggle saving
        #print(' 8c')
        i=10
        for j in range(10):  # go thorugh all button_frames, and show/hide them as required
            if len(self.button_frames[i][j].winfo_children()) == 0:
                self.button_frames[i][j].pack_forget()
            else:
                self.button_frames[i][j].pack(
                    side=tkinter.LEFT, anchor=tkinter.E)
            #self.update()
            #time.sleep(0.01)
        #print(' 8d')
        try:
            if len(used_tabs) < 2:
                self.notebook.pack_forget()
                #time.sleep(0.01)
                #self.update()
                self.style.layout('TNotebook.Tab', [])
                #self.update()
                self.notebook.pack(side=tkinter.TOP, anchor=tkinter.E,fill=tkinter.BOTH)
                #print(' 8d2')
                #time.sleep(0.01)
                #self.update()
            else:
                self.notebook.pack_forget()
                #time.sleep(0.01)
                #self.update()
                self.style.layout('TNotebook.Tab', self.notebook_style)
                self.notebook.pack(side=tkinter.TOP, anchor=tkinter.E,fill=tkinter.BOTH)
                #time.sleep(0.01)
                #self.update()
        except Exception as e:
            print(e)
        #time.sleep(0.01)
        #self.update()
        #print(' 8e')

        for i in range(1,10):
            if i not in used_tabs:
                l=tkinter.Label(self.notebook_tabs[i],text='')
                l.pack()
                #self.update()
                #time.sleep(0.01)
        #self.update()
        for i in range(1,10):
            if i not in used_tabs:
                self.notebook_tabs[i].winfo_children()[-1].destroy()
                #self.update()

        #print(' 8f')
        if len(used_tabs)==0:
            self.notebook.add(
                self.notebook_tabs[0], text=self.notebook_tab_labels[i])
            l=tkinter.Label(self.notebook_tabs[0],text='')
            l.pack()
            self.notebook_tabs[0].winfo_children()[-1].destroy()
            #self.update()
    def buttons_to_map(self):
        map={}
        for key in self.buttons:
            try:
                map[key]=self.buttons[key].get()
                try:
                    type=self.buttons[key].button_return_type
                    if 'txt' in type:
                        if 'int' in type:
                            if 'range' in type:
                                try:
                                    map[key]=fns.int_string_to_list(map[key])
                                except:
                                    print('error when casting string in field',key,'to list of integers:')
                                    print(map[key])
                            else:
                                try:
                                    map[key]=int(map[key])
                                except:
                                    print('error when casting string in field',key,'to int')
                                    print(map[key])
                        if 'float' in type:
                            if 'range' in type:
                                try:
                                    map[key]=fns.float_string_to_list(map[key])
                                except:
                                    print('error when casting string in field',key,'to list of floats:')
                                    print(map[key])
                            else:
                                try:
                                    map[key]=float(map[key])
                                except:
                                    print('error when casting string in field',key,'to float')
                                    print(map[key])
                    elif type=='radio:text':
                        try:
                            map[key]=self.buttons[key].button_return_texts[map[key]]
                        except:
                            print('error when getting state of radio button',key,'as text')
                            print(map[key])
                except:
                    None #is a check button
            except:
                map[key]=self.buttons[key] #is a click buttons
        map['save_check']=self.save_check_var.get()
        map['save_filename']=self.name_field.get()
        self.map=map
        return map

    def toggle_threading(self):
        if self.threading_check_var.get():
            self.thread_label.pack(side=tkinter.LEFT)
            self.stop.pack(side=tkinter.LEFT)
        else:
            self.thread_label.pack_forget()
            self.stop.pack_forget()

    def stopbuttonpress(self):
        # create thread that stops other threads
        stop_thread_ = stop_thread(self)
        stop_thread_.start()

    # function that runs when the figure element is clicked.
    def callbackClick(self, event):

        if hasattr(self,'run'):
            # if the first measurement currently shown has a function called 'callbackClick', call it
            if hasattr(self.run, 'callbackClick'):
                self.run.callbackClick(self, event)
            else:  # otherwise just print coordinates
                fns.fnsCallBackClick(self, event)

    # on window close
    def on_delete(self):
        self.quit_flag = 1
        #time.sleep(0.02)
        self.destroy()
        root.quit()
        root.destroy()
        sys.exit()
    # when run button is pressed:

    def runbuttonpress(self):  # run
        gc.collect() #collect garbage to free memory from last run
        for thread in threading.enumerate():
            if type(thread) == run_as_thread:
                # already running -> ignore
                return
        #empty queues
        while True:
            try:
                dummy = self.send_queue.get(False)  # doesn't block
                #eprint(dummy)
            except:
                break
        while True:
            try:
                dummy=self.return_queue.get(False)  # doesn't block
                #eprint(dummy)
            except:
                break
        #close all Figures: if the previous run was aborted, there may be figures that are still present and taking up memoryself.
        # in practice the program will crash if plt.close(figure) is called on them. But we can call fig.cla()
        figures=[manager.canvas.figure for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        for figure in figures:
            if not figure == self.figure:
                figure.clf()
                #plt.close(figure) #this line will cause the program to crash if a previous run has been stopped with the stop button

        # reload all modules. Only usefull if the code in one of the modules has been changed since last time.
        self.import_modules(force=False)
        self.files=self.nav.get_paths_of_selected_items()
        if self.threading_check_var.get():
            importThread = run_as_thread(self)
            importThread.start()
            return  # we are done here, the child thread handles all the rest, so that we do not mess upp the order
        else:
            temp = self.matplotlib_plot_num
            self.matplotlib_plot_num = 2
            run_module(self)
            self.matplotlib_plot_num = temp  # normally starts at #10, #2 is reserved for when not using threading
            # increment plot number if the default save name is used, and savecheck is checked
            return

    def show_data_tree(self):
        if self.data_tree_frame.winfo_ismapped():
            self.data_tree_frame.pack_forget()
        else:
            self.data_tree_frame.pack(fill=tkinter.BOTH, expand=True,side=tkinter.LEFT, anchor=tkinter.E)
            self.data_tree.rebuild()

    def execute_command(self, arg=None):
        command = self.console_intput.get('1.0', 'end')
        if hasattr(self,'command_history'):
            self.command_history.append(command)
        else:
            self.command_history=['',command]
        self.current_line_in_history=0
        sys.stdout = Std_redirector(self.console_output_box)
        try:
            exec(command)
        except Exception as e:
            traceback.print_exc()
            print(e)
        self.after(10, self.console_intput.delete,'1.0', 'end' )
        sys.stdout = Std_redirector(self.output_box)

        #sys.stderr=self.stderr
    def set_pervious_command(self, arg=None):
        if hasattr(self,'command_history'):
            self.current_line_in_history=(self.current_line_in_history-1)%len(self.command_history)
            self.console_intput.delete('1.0', 'end' )
            self.console_intput.insert('1.0',self.command_history[self.current_line_in_history])
    def set_next_command(self, arg=None):
        if hasattr(self,'command_history'):
            self.current_line_in_history=(self.current_line_in_history+1)%len(self.command_history)
            self.console_intput.delete('1.0', 'end' )
            self.console_intput.insert('1.0',self.command_history[self.current_line_in_history])

    def count_thread(self):
        count = threading.activeCount()
        threads=0
        for thread in threading.enumerate():
            if not (thread==self.main_thread  or thread.__class__==threading._DummyThread):
                threads+=1
        if not self.old_threads==threads:
            self.thread_label.config(text='Threads: '+str(threads))
            self.old_threads=threads
        self.after(100,self.count_thread)
# lists thread in main window:


# import measurement as thread:
def get_row_tab(b):
    if 'row' in b:
        row=b['row']
    else:
        row=0
    if 'tab' in b:
        tab=b['tab']
    else:
        tab=0
    return row, tab

class run_as_thread (threading.Thread):
    def __init__(self, frame):
        threading.Thread.__init__(self)
        threading.Thread.setDaemon(self, True)
        self.frame = frame

    def run(self):
        run_module(self.frame)

    def _get_my_tid(self):
        if not self.isAlive():
            raise threading.ThreadError("the thread is not active")
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for tid, tobj in threading._active.items():
            if tobj is self:
                self._thread_id = tid
                return tid
        raise AssertionError("could not determine the thread's id")

    def raiseExc(self, exctype):  # used to kill the thread if the stop button is pressed
        """Raises the given exception type in the context of this thread.

        If the thread is busy in a system call (time.sleep(),
        socket.accept(), ...), the exception is simply ignored.

        If you are sure that your exception should terminate the thread,
        one way to ensure that it works is:

            t = ThreadWithExc( ... )
            ...
            t.raiseExc( SomeException )
            while t.isAlive():
                time.sleep( 0.1 )
                t.raiseExc( SomeException )

        If the exception is to be caught by the thread, you need a way to
        check that your thread has caught it.

        CAREFUL : this function is executed in the context of the
        caller thread, to raise an excpetion in the context of the
        thread represented by this instance.
        """
        _async_raise(self._get_my_tid(), exctype)

def run_module(frame):
    files = frame.files  # if no files are selected, this will recall last used set of files
    frame.rimt.rimt(frame.update)

    for c in frame.recognised_files[frame.current_filetype]:
        try:
            button_map=frame.buttons_to_map()
            frame.run = c(frame.figure,files, frame, button_map)
            frame.run.run()
        except Exception as e:
            if not "'moduleClass' object has no attribute 'run'" == str(e):
                errortext=traceback.format_exc()
                exceptiont_to_fig(frame,errortext)
    update_figure_and_filename(frame) #separated to function because it needs to run in the main thread by using the decorator @fns.rimt

@fns.rimt
def update_figure_and_filename(frame): #separated to function because it needs to run in the main thread by using the decorator @fns.rimt
    frame.update()
    frame.figure.canvas.draw()
    frame.data_tree.rebuild()
    # increment plot number if the default save name is used, and savecheck is checked
    picname = frame.name_field.get()
    if picname == frame.default_filename+str(frame.plot_number):
        if frame.save_check_var.get():
            frame.name_field_string.set(
                frame.default_filename+str(frame.plot_number))
            frame.plot_number += 1
            frame.name_field_string.set(
                frame.default_filename+str(frame.plot_number))

@fns.rimt
def exceptiont_to_fig(frame,errortext):
    if hasattr(frame,'error_ax'):
        if frame.error_ax in frame.figure.axes:
            frame.error_ax.cla()
        else:
            frame.error_ax=frame.figure.add_axes([0,0,0,0])
    else:
        frame.error_ax=frame.figure.add_axes([0,0,0,0])
    eprint(errortext)
    frame.error_ax.patch.set_alpha(0.5)
    frame.error_ax.text(0,0,errortext)
    frame.figure.canvas.draw()
    frame.update()


class stop_thread (threading.Thread):
    def __init__(self, frame):
        threading.Thread.__init__(self)
        threading.Thread.setDaemon(self, True)
        self.frame = frame

    def run(self):
        self.frame.stop_flag = 1
        time.sleep(0.1)
        for thread in threading.enumerate():
            if type(thread) == run_as_thread:
                while thread.isAlive():
                    thread.raiseExc(KeyboardInterrupt)
                    time.sleep(0.1)
                    self.frame.return_queue.put('None')
                    self.frame.send_queue.put('stop')
                    time.sleep(0.1)
        #empty queues:
        while True:
            try:
                self.frame.send_queue.get(False)  # doesn't block
            except:
                break
        while True:
            try:
                self.frame.return_queue.get(False)  # doesn't block
            except:
                break
        self.frame.pools = []
        self.frame.stop_flag = 0
        self.frame.parent.after(10, update_figure, self.frame.parent, self.frame.send_queue, self.frame.return_queue,self.frame)

# used to when killing thread:


def _async_raise(tid, exctype):
    '''Raises an exception in the threads with id tid'''
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # "if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")
try: # the syntax is changed between python 3 and python 3.6 # NavigationToolbar2Tk
    class CustomToolbar(matplotlib.backends.backend_tkagg.NavigationToolbar2Tk):
        def __init__(self, canvas_, parent_):
            self.toolitems = (
                ('Home', 'Reset orginal view', 'home', 'home'),
                # ('Back', 'consectetuer adipiscing elit', 'back', 'back'),
                # ('Forward', 'sed diam nonummy nibh euismod', 'forward', 'forward'),
                (None, None, None, None),
                ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
                ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
                (None, None, None, None),
                ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
                ('Save', 'Save', 'filesave', 'save_figure'),
            )
            matplotlib.backends.backend_tkagg.NavigationToolbar2Tk.__init__(
                self, canvas_, parent_)
except: # the syntax is changed between python 3 and python 3.6 # NavigationToolbar2TkAgg
    class CustomToolbar(matplotlib.backends.backend_tkagg.NavigationToolbar2TkAgg):
        def __init__(self, canvas_, parent_):
            self.toolitems = (
                ('Home', 'Reset orginal view', 'home', 'home'),
                # ('Back', 'consectetuer adipiscing elit', 'back', 'back'),
                # ('Forward', 'sed diam nonummy nibh euismod', 'forward', 'forward'),
                (None, None, None, None),
                ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
                ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
                (None, None, None, None),
                ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
                ('Save', 'Save', 'filesave', 'save_figure'),
            )
            matplotlib.backends.backend_tkagg.NavigationToolbar2TkAgg.__init__(
                self, canvas_, parent_)

# redirects stdout to the main window


class Std_redirector(object):
    def __init__(self, widget):
        self.widget = widget
        self.flush = sys.stdout.flush
    @fns.rimt
    def write(self, string):
        self.widget.insert(tkinter.END, string)
        self.widget.see(tkinter.END)
# main function to open window


def main():
    global root

    send_queue = queue.Queue()
    return_queue = queue.Queue()
    fns.return_queue = return_queue
    fns.send_queue = send_queue
    fns.main_thread = threading.currentThread()
    root = tkinter.Tk()
    root.geometry("1200x800+300+300")

    press_alt_label = tkinter.Label(
        root, text='Press the "Alt" key to continue', background="#FFFFFF")
    press_alt_label.pack(side=tkinter.LEFT)
    root.update()
    settings=make_or_get_settings()
    app = mainFrame(root,settings)

    #setattr(app,'update', fns.rimt(getattr(app,'update')))
    app.send_queue = send_queue
    app.return_queue = return_queue
    app.rimt = rimt(app.send_queue, app.return_queue)
    root.protocol("WM_DELETE_WINDOW", app.on_delete)
    root.after(1, update_figure, root, app.send_queue, app.return_queue,app)
    root.update()
    press_alt_label.pack_forget()
    root.mainloop()

def make_or_get_settings():
    settings=get_default_settings()
    try:
        with open('settings','r') as settingsfile:
            for line in settingsfile:
                key=str(line.split(' ')[0])
                if key in settings:
                    value=line.split(' ')[1].strip()
                    if type(settings[key]) is int:
                        value=int(value)
                    settings[key]=value
    except:
        if os.path.isfile('settings'):
            print('error reading settings file')
            return get_default_settings()
        else:
            with open('settings','w') as settingsfile:
                for key in settings:
                    settingsfile.write(key+' '+str(settings[key])+'\n')
    return settings

def get_default_settings():
    default_settings={'save_path':'output/%date%/plot-',
    'save_check':0,
    'list_sort_type':0,
    'list_sort_direction':0,
    'threading':0,
    'reload_check':0,
    'root_folder':'%Path_of_program%/'
    }
    return default_settings

    #try:
    #    with open()'settings',r) as settingsfile:

def update_figure(root, send_queue, return_queue,frame):
    #root.update()
    for i in [0]:
        try:
            callback = send_queue.get(False)  # doesn't block
        except:  # queue.Empty raised when queue is empty (python3.7)
            break
        if callback=='stop': # if the stop button is pressed the update_figure loop should end
            return
        try:
            return_parameters = callback()
            return_queue.put(return_parameters)
        except Exception as e:
            return_parameters = None
            traceback.print_exc()
            eprint(e)
            try:
                frame.stopbuttonpress()
                errortext=traceback.format_exc()
                exceptiont_to_fig(frame,errortext)
            except:
                None
    root.after(3, update_figure, root, send_queue, return_queue,frame)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class rimt():
    def __init__(self, send_queue, return_queue):
        self.send_queue = send_queue
        self.return_queue = return_queue
        self.main_thread=threading.currentThread()

    def rimt(self, function, *args, **kwargs):
        if threading.currentThread() == self.main_thread:
            return function(*args, **kwargs)
        else:
            self.send_queue.put(functools.partial(function, *args, **kwargs))
            return_parameters = self.return_queue.get(True)  # blocks until an item is available
        return return_parameters


    # :)
if __name__ == '__main__':
    main()
