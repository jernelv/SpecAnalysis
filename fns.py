#!/usr/bin/python

# matplotlib.rc('text', usetex=True)
# matplotlib.rc('text.latex', preamble=r'\usepackage{upgreek}')
import functools
import multiprocessing
import time
import threading
import sys
import matplotlib
import numpy as np
import datetime

def rimt(function):
    def rimt_this(*args, **kwargs):
        global list_of_calls
        list_of_calls.append(function)
        global send_queue
        global return_queue
        global main_thread
        if threading.currentThread() == main_thread:
            return function(*args, **kwargs)
        else:
            send_queue.put(functools.partial(function, *args, **kwargs))
            # blocks until an item is available
            return_parameters = return_queue.get(True)
            return return_parameters
    return rimt_this

def fnsCallBackClick(frame, event):
    print("clicked at", event.xdata, event.ydata)

def is_twin(ax_list,ax):
    twins=ax.get_shared_x_axes().get_siblings(ax)
    if len(twins)>1:
        for twin in twins:
            if twin in ax_list:
                #the current ax is assumed to be the a twin
                return True
    return False

@rimt
def move_all_plots(figure,fig_per_row,max_plots=-1,extra=0,direction='right'):
    if not max_plots==-1 and fig_per_row>max_plots:
        fig_per_row=max_plots
    ax_list = []
    cax_list = []
    for ax in figure.axes:
        if is_twin(ax_list,ax):
            continue
        if len(ax.artists) == 0 or not matplotlib.patches.Polygon == type(ax.artists[0]):
            ax_list.append(ax)
            cax_list.append(None)
        else:
            cax_list[-1] = ax
    if not max_plots==-1 and max_plots<len(ax_list)+extra+1:
        start=len(ax_list)+extra-max_plots
        for ax in ax_list[:start]:
            figure.delaxes(ax)
        for ax in cax_list[:start]:
            if not ax==None:
                figure.delaxes(ax)
        ax_list=ax_list[start:]
        cax_list=cax_list[start:]
    rows = max((len(ax_list)+extra+fig_per_row-1)//fig_per_row, 1)
    if direction=='right':
        rowstep = 0.96/rows
        colstep = 0.96/fig_per_row
    else:
        rowstep = 0.96/fig_per_row
        colstep = 0.96/rows
    if len(ax_list) > 0:
        for i, ax in enumerate(ax_list):
            if direction=='right':
                row = i//fig_per_row
                col = i % fig_per_row
            else:
                col = i//fig_per_row
                row = i % fig_per_row
            pos = [0.04+colstep*(col+0.125), 1-rowstep *
                   (row+1-0.15), colstep*0.75, rowstep*(1-0.28)]
            ax.set_position(pos, which='original')
            if not cax_list[i] == None:
                move_cbar(ax, cax_list[i])
    else:
        i = -1
    return i,rowstep,colstep

def add_axis(figure,fig_per_row=4,max_plots=-1,direction='right'):
    if not max_plots==-1 and fig_per_row>max_plots:
        fig_per_row=max_plots
    #fig_per_row = int(self.buttons['PFMnumFigPerRow'].get())
    i,rowstep,colstep=move_all_plots(figure,fig_per_row,max_plots,extra=1,direction=direction)
    if direction=='right':
        row = (i+1)//fig_per_row
        col = (i+1) % fig_per_row
    else:
        col = (i+1)//fig_per_row
        row = (i+1) % fig_per_row
    pos = [0.04+colstep*(col+0.125), 1-rowstep *
           (row+1-0.15), colstep*0.75, rowstep*(1-0.28)]
    now = datetime.datetime.now()
    label = now.strftime('%m:%d:%H:%M:%S.%f')+str(np.random.rand())
    newax = figure.add_axes(pos,label=label) # a unique label is required for each axis, because add_axes() will otherwise return a previous ax that was created with the same position, regardless if that ax was later moved
    return newax

class Pool(object):
    # this class is designed to be used like 'multiprocessing.Pool()', but with the added functionality that the pool will stop if frame.stopflag is set to 1
    frame = None

    def __init__(self, n):
        self.p = multiprocessing.Pool(n, maxtasksperchild=1)
        # self.frame.pools.append(self.p) # this adds it to array so that the pool is stored for later use, is this needed?

    def map(self, fun, par):
        i = 0
        maxtries = 10
        while i < maxtries:
            try:
                r = self.p.map_async(fun, par)
                while True:
                    if self.frame.stop_flag == 1:
                        self.p.terminate()  # stops all child processes
                        self.p.join()  # joins them
                        sys.exit()  # stops current thread
                        return []
                    if not r.ready():
                        time.sleep(0.05)
                    else:
                        self.p.terminate()  # stops all child processes
                        self.p.join()
                        return r.get(0)
            except:
                if self.frame.stop_flag == 0:
                    print('pool failed (attempt '+str(i) +
                          ' of ' + str(maxtries)+' )')
                    i += 1

    def close(self):
        self.p.close()
        return



main_thread = None
return_queue = None
send_queue = None
list_of_calls=[]


def print_list_of_calls():
    global list_of_calls
    return list_of_calls

@rimt
def move_cbar(parents, cax, location=None, orientation=None, fraction=0.15,
              shrink=1.0, aspect=20, **kw):
    '''
    Resize and reposition parent(s) axes and cax. This code is an excerpt
        from matplotlib.colorbar.make_axes, adapted to move the cax rather
        than create a new cax
    Keyword arguments may include the following (with defaults):

        location : [None|'left'|'right'|'top'|'bottom']
            The position, relative to **parents**, where the colorbar axes
            should be created. If None, the value will either come from the
            given ``orientation``, else it will default to 'right'.
        orientation :  [None|'vertical'|'horizontal']
            The orientation of the colorbar. Typically, this keyword shouldn't
            be used, as it can be derived from the ``location`` keyword.
    %s

    Returns (none)
    '''
    locations = ["left", "right", "top", "bottom"]
    if orientation is not None and location is not None:
        raise TypeError('position and orientation are mutually exclusive. '
                        'Consider setting the position to any of {}'
                        .format(', '.join(locations)))

    # provide a default location
    if location is None and orientation is None:
        location = 'right'

    # allow the user to not specify the location by specifying the
    # orientation instead
    if location is None:

        location = 'right' if orientation == 'vertical' else 'bottom'

    if location not in locations:
        raise ValueError('Invalid colorbar location. Must be one '
                         'of %s' % ', '.join(locations))

    default_location_settings = {'left':   {'anchor': (1.0, 0.5),
                                            'panchor': (0.0, 0.5),
                                            'pad': 0.10,
                                            'orientation': 'vertical'},
                                 'right':  {'anchor': (0.0, 0.5),
                                            'panchor': (1.0, 0.5),
                                            'pad': 0.05,
                                            'orientation': 'vertical'},
                                 'top':    {'anchor': (0.5, 0.0),
                                            'panchor': (0.5, 1.0),
                                            'pad': 0.05,
                                            'orientation': 'horizontal'},
                                 'bottom': {'anchor': (0.5, 1.0),
                                            'panchor': (0.5, 0.0),
                                            'pad': 0.15,  # backwards compat
                                            'orientation': 'horizontal'},
                                 }

    loc_settings = default_location_settings[location]

    # put appropriate values into the kw dict for passing back to
    # the Colorbar class
    kw['orientation'] = loc_settings['orientation']
    kw['ticklocation'] = location

    anchor = kw.pop('anchor', loc_settings['anchor'])
    parent_anchor = kw.pop('panchor', loc_settings['panchor'])

    parents_iterable = matplotlib.cbook.iterable(parents)
    # turn parents into a list if it is not already. We do this w/ np
    # because `plt.subplots` can return an ndarray and is natural to
    # pass to `colorbar`.
    parents = np.atleast_1d(parents).ravel()

    # check if using constrained_layout:
    try:
        gs = parents[0].get_subplotspec().get_gridspec()
        using_constrained_layout = (gs._layoutbox is not None)
    except AttributeError:
        using_constrained_layout = False

    # defaults are not appropriate for constrained_layout:
    pad0 = loc_settings['pad']
    if using_constrained_layout:
        pad0 = 0.02
    pad = kw.pop('pad', pad0)

    fig = parents[0].get_figure()
    if not all(fig is ax.get_figure() for ax in parents):
        raise ValueError('Unable to create a colorbar axes as not all '
                         'parents share the same figure.')

    # take a bounding box around all of the given axes
    parents_bbox = matplotlib.transforms.Bbox.union(
        [ax.get_position(original=True).frozen() for ax in parents])

    pb = parents_bbox
    if location in ('left', 'right'):
        if location == 'left':
            pbcb, _, pb1 = pb.splitx(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splitx(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(1.0, shrink).anchored(anchor, pbcb)
    else:
        if location == 'bottom':
            pbcb, _, pb1 = pb.splity(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splity(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(shrink, 1.0).anchored(anchor, pbcb)

        # define the aspect ratio in terms of y's per x rather than x's per y
        aspect = 1.0 / aspect

    # define a transform which takes us from old axes coordinates to
    # new axes coordinates
    shrinking_trans = matplotlib.transforms.BboxTransform(parents_bbox, pb1)

    # transform each of the axes in parents using the new transform
    for ax in parents:
        new_posn = shrinking_trans.transform(ax.get_position(original=True))
        new_posn = matplotlib.transforms.Bbox(new_posn)
        ax._set_position(new_posn)
        if parent_anchor is not False:
            ax.set_anchor(parent_anchor)
    cax.set_position(pbcb)

def decorate_matplotlib():
    functions_to_wrap=[
    [matplotlib.axes.Axes,'text'],
    [matplotlib.backends.backend_tkagg.FigureCanvasTkAgg,'draw']
    ]
    functions_not_to_vrap=[
    'rimt.<locals>.rimt_this',
    'Axes._pcolorargs',
    'Axes._remove_legend',
    'Axes._pcolorargs',
    'Axes._quiver_units',
    'Axes._pcolorargs',
    'Figure.__str__',
    'Figure.__repr__',
    'Figure.__init__',
    'Figure._repr_html_',
    'Figure._get_axes',
    'Figure._get_dpi',
    'Figure._set_dpi',
    'Figure._remove_ax',
    'Figure._make_key',
    'Figure._set_artist_props',
    'Figure._gci',
    'Figure.__getstate__',
    'Figure.__setstate__',
    'Figure.__setstate__',
    'Figure.__setstate__',
    'Figure.__setstate__',
    ]
    for key in matplotlib.axes.Axes.__dict__:
        functions_to_wrap.append([matplotlib.axes.Axes,key])
    for key in matplotlib.figure.Figure.__dict__:
        functions_to_wrap.append([matplotlib.figure.Figure,key])
    for key in matplotlib.pyplot.__dict__:
        functions_to_wrap.append([matplotlib.pyplot,key])
    for function in functions_to_wrap:
        if '<function ' in str(getattr(function[0], function[1])):
            if not str(getattr(function[0], function[1])).split('<function ')[1].split(' at ')[0] in functions_not_to_vrap:
                #print('wrapping',str(getattr(function[0], function[1])),str(getattr(function[0], function[1])).split('<function ')[1].split(' at ')[0])
                setattr(function[0], function[1], rimt(getattr(function[0], function[1])))
            #else:
            #    print('################### not wrapping' ,str(getattr(function[0], function[1])))
    return

def int_string_to_list(input):
    ll=[]
    steps=input.split(',')
    for step in steps:
        is_range=step.split(':')
        if len(is_range)==3:
            start=int(is_range[0])
            stop=int(is_range[1])
            step=int(is_range[2])
            ll+=list(range(start,stop,step))
        elif len(is_range)==2:
            start=int(is_range[0])
            stop=int(is_range[1])
            ll+=list(range(start,stop))
        else: #len(is_range)==1:
            val=int(is_range[0])
            ll+=[val]
    return ll
def float_string_to_list(input):
    ll=[]
    steps=input.split(',')
    for step in steps:
        is_range=step.split(':')
        if len(is_range)==3:
            start=float(is_range[0])
            stop=float(is_range[1])
            step=float(is_range[2])
            ll+=list(np.arange(start,stop,step))
        elif len(is_range)==2:
            start=float(is_range[0])
            stop=float(is_range[1])
            ll+=list(np.arange(start,stop))
        else: #len(is_range)==1:
            val=float(is_range[0])
            ll+=[val]
    return ll
