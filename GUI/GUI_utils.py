from . import Path, Tk, ttk, StringVar, IntVar, BooleanVar, filedialog, FigureCanvasTkAgg, NavigationToolbar2Tk, plt

def open_dirGUI(init_path):
    # Creates a GUI for selecting folder path and returns the folder
    # In:
    #   init_path:                          str, name of the initial search path
    # Out:
    #   final_path:                         str, the path which has been selected by the user

    root = Tk()
    root.withdraw()
    final_path = Path(filedialog.askdirectory(initialdir=init_path))
    root.destroy()
    return final_path

def open_fileGUI(init_path, filetypes=(("all files", "*.*"), )):
    # Creates a GUI for selecting folder path and returns the file
    # In:
    #   init_path:                          str, name of the initial search path
    # Out:
    #   final_path:                         str, the path which has been selected by the user

    root = Tk()
    root.withdraw()
    final_path = Path(filedialog.askopenfilename(initialdir=init_path, filetypes=filetypes))
    root.destroy()
    return final_path

def show_data_tk(frame, data, label, pack={'row':0, 'column':0}):
    # 
    # In:
    #   frame:                  tkinter frame object to pack display objects or a list containing created widgets
    #   data:                   data to show
    #   label:                  label to data
    #   pack:                   grid packing arguments for the display frame

    if isinstance(frame, tuple):
        fig, ax, canvas, toolbar = frame
    else:
        data_dims = len(data.shape)

        fig, ax = plt.subplots()
        frame = ttk.Frame(frame)
        canvas = FigureCanvasTkAgg(fig, frame) 
        canvas.get_tk_widget().pack()
        frame.grid(**pack)

        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.pack()

    ax.clear()
    ax.imshow(data)
    canvas.draw()
    toolbar.update()

    return (fig, ax, canvas, toolbar)

def pack_object(obj, function, values):
    # Packs objects with tkinter grid and pack functions
    # In:
    #   obj:                     Tkinter object, root or frame
    #   function:                str, packing function name (pack, grid)
    #   values:                  tuple, list or dict containing packing attrs, example: {'row':0, 'column':0}
 
    if hasattr(obj, function):
        pack_func = getattr(obj, function)
        if values is not None:
            if isinstance(values, dict):
                pack_func(**values)
            elif isinstance(values, (tuple, list)):
                pack_func(*values)
        else:
            pack_func()
    else:
        print("Must have 'grid' or 'pack' key value pair, where value holds packing attributes!")
    
def get_packingdata(values):
    # Packing params parser
    # In:
    #   values:                  dict, key = packing function, values = inputs to packing function
    # Out:
    #   tuple:                   function = name of the packing function, pack_values = pack function inputs
    
    if 'grid' in values.keys():
        function = 'grid'
        pack_values = values[function]
    elif 'pack' in values.keys():
        function = 'pack'
        pack_values = values[function]
    return (function, pack_values)

def pack(obj, values=None):
    # Parse values and call packing function
    # In:
    #   obj:                     ?
    #   values:                  ?
    # Out:
    #   tuple:                   function = name of the packing function, pack_values = pack function inputs

    if values is None:
        values = obj
        obj = obj['ref']
    function, pack_values = get_packingdata(values)
    pack_object(obj, function, pack_values)
    return (function, pack_values)

def hide_widgets(widget):
    # Hides the input widget from the window
    # In:
    #   widget:                  Tkinter widget

    obj = widget['ref']
    if 'grid' in widget.keys():
        obj.grid_forget()
    elif 'pack' in widget.keys():
        obj.pack_forget()

def create_frame(pack_to, values):
    # Creates tkinter frame object
    # In:
    #   pack_to:                 Tkinter master, where frame is packed
    # Out:
    #   dict:                    ref key contains the frame object and function key contains its attributes

    frame = ttk.Frame(pack_to)
    function, attrs = pack(frame, values)
    return {'ref':frame, function:attrs}

def handle_frame(values, tkclass=None, buildmap=None):
    # Handles frame building
    # In:
    #   values:                  dict, contains the widget and packing info
    #   tkclass:                 Class where widgets are build
    #   buildmap:                dict, previously build tkinter structure
    #   * tkclass or buildmap must be fed when function is called, if both are fed buildmap is used

    if 'master' in values.keys():
        master = values['master']
        pack_to = None
        if buildmap is not None:
            if master in buildmap.keys():
                pack_to = buildmap[master]

        if tkclass is not None and pack_to is None:
            #print(master)
            if hasattr(tkclass, master):
                pack_to = getattr(tkclass, master)
            else:
                if hasattr(tkclass, 'widgets'):
                    if master in tkclass.widgets.keys():
                        pack_to = tkclass.widgets[master]['ref']

        return create_frame(pack_to, values)
    else:
        print("Values must have 'master' key value pair, where value is the name of the object where to pack the frame!")

def handle_variable(variable):
    # Variable handling if tkinter Var is used
    # In:
    #   variable:                str, int or bool
    # Out:
    #   str, int or bool tkinter Var

    if isinstance(variable, str):
        return StringVar()
    elif isinstance(variable, int):
        return IntVar()
    elif isinstance(variable, bool):
        return BooleanVar()
    else:
        print("No action for ", type(variable), variable)

def handle_optionmenus(tkclass, name, values, master):
    # Handle optionmenu building
    # In:
    #   tkclass:                 class where tkinter is build
    #   name:                    str, name for the widget
    #   values:                  dict, containing options information
    #   master:                  tkinter object where widgets are packed
    # Out:
    #   tuple:                   containing (dict) optionmenu ref and packing info and tkinter Var object

    variable = None
    if 'variable' in values.keys():
        variable = values['variable']
        variable = handle_variable(variable)
    
    if 'options' in values.keys():
        options = values['options']
        if variable is None:
            variable = handle_variable(options[0])

    optionmenu = ttk.OptionMenu(master, variable, *options)
    function, pack_data = pack(optionmenu, values)

    if 'set' in values.keys():
        variable.set(values['set'])

    return ({'ref':optionmenu, function:pack_data}, variable)

def handle_buttons(tkclass, name, values, master):
    # Handle button building
    # In:
    #   tkclass:                 class where tkinter is build
    #   name:                    str, name for the widget
    #   values:                  dict, containing button information
    #   master:                  tkinter object where widgets are packed
    # Out:
    #   dict:                    button widget ref and packing info
    
    if 'command' in values.keys():
        command = getattr(tkclass, values['command'])
        button = ttk.Button(master, text=values['text'], command=lambda:command())
    else:
        button = ttk.Button(master, text=values['text'])

    function, pack_data = pack(button, values)
    
    return {'ref':button, function:pack_data}

def handle_spinboxes(tkclass, name, values, master):
    # Handle spinbox building
    # In:
    #   tkclass:                 class where tkinter is build
    #   name:                    str, name for the widget
    #   values:                  dict, containing spinbox information
    #   master:                  tkinter object where widgets are packed
    # Out:
    #   dict:                    spinbox widget ref and packing info
    
    attributes = {}
    for key, value in values.items():
        if key in ['from_', 'to', 'format', 'increment']:
            attributes[key] = value
    spinbox = ttk.Spinbox(master, **attributes)
    function, pack_data = pack(spinbox, values)

    if 'set' in values.keys():
        spinbox.set(values['set'])

    return {'ref':spinbox, function:pack_data}

def handle_labels(tkclass, name, values, master):
    # Handle label building
    # In:
    #   tkclass:                 class where tkinter is build
    #   name:                    str, name for the widget
    #   values:                  dict, containing label information
    #   master:                  tkinter object where widgets are packed
    # Out:
    #   dict:                    label widget ref and packing info and tkinter Var object
    
    label = ttk.Label(master, text=values['text'])
    function, pack_data = pack(label, values)

    return {'ref':label, function:pack_data}

def handle_entries(tkclass, name, values, master):
    # Handle entry building
    # In:
    #   tkclass:                 class where tkinter is build
    #   name:                    str, name for the widget
    #   values:                  dict, containing entry information
    #   master:                  tkinter object where widgets are packed
    # Out:
    #   dict:                    entry widget ref and packing info and tkinter Var object
    
    entry = ttk.Entry(master)
    function, pack_data = pack(entry, values)

    return {'ref':entry, function:pack_data}


# Function mapping
# In GUI_config key containing information for all widgets in a class, the key name must map to one of these
handle_widget_function_map = {
        'optionmenus':handle_optionmenus,
        'buttons':handle_buttons,
        'spinboxes':handle_spinboxes,
        'labels':handle_labels,
        'entries':handle_entries,
        }

def handle_widgets(tkclass, name, values, master):
    # Handle widget building

    widgetmap = {}
    for wname, wvalues in values.items():
        widgets = handle_widget_function_map[name](tkclass, wname, wvalues, master)
        if isinstance(widgets, tuple):
            widgets, widgetvars = widgets
            widgetmap[wname+'_var'] = widgetvars
        
        widgetmap[wname] = widgets

    return widgetmap

def add_to_frame(tkclass, frame, add, wtype):
    # Add a widget to a frame

    if 'ref' in frame.keys():
        frame = frame['ref']
    for key, value in add.items():
        widgetmap = handle_widget_function_map[wtype](tkclass, key, value, frame)

    return widgetmap

def build_blueprint(ui, blueprint):
    # The main builder function, which parses the config dict and builds the window

    buildmap = {}
    def builder(name, values):
        for widget_name, widget_values in values.items():
            if widget_name == 'master':
                print("Creating frame ", name)
                buildmap[name] = handle_frame(values, ui)
            elif 'master' in widget_values:
                builder(str(name+'_'+widget_name), widget_values)
            elif widget_name in handle_widget_function_map.keys():
                print("Creating widgets ", widget_name)
                buildmap[name+"_"+widget_name] = handle_widgets(ui, widget_name, widget_values, buildmap[name]['ref'])
        
    #print(blueprint)
    for key, values in blueprint.items():
        print("Top level: ", key, values)
        builder(key, values)

    return buildmap
