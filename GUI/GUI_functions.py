from . import partial, Tk, ttk, filedialog, StringVar, Path

#TODO
# Move to MYSQL selector folder
class define_relation:
    # Quick and dirty GUI for selecting a relational database relations

    def __init__(self, columns1, columns2):
        root = Tk()
    
        def create_rbs(cols, row, var):
            for i, column in enumerate(cols):
                rb = ttk.Radiobutton(
                    root, 
                    text=column, 
                    variable=var, 
                    value=column,
                    )
                rb.grid(column=i+1, row=row)
    
        self.relation_1 = StringVar()
        self.relation_2 = StringVar()
        create_rbs(columns1, 0, self.relation_1)
        create_rbs(columns2, 1, self.relation_2)

        def quit():
            root.destroy()
            exit()

        def select():
            if not self.relation_1.get() or not self.relation_2.get():
                print("Choose both relations!")
            else:
                root.destroy()

        ttk.Button(root, text="Select", command=select).grid(column=0, row=2)
        ttk.Button(root, text="Quit", command=quit).grid(column=0, row=3)
        root.mainloop()

#TODO
# Move to its own folder
class Mysql_data_selector:
    # Experimental data selector GUI

    def __init__(self, connector):
        self.root = Tk()
        self.connector = connector
        self.table_selector()
        self.root.mainloop()

    def table_selector(self):
        table_frame = ttk.Frame(self.root)
        self.table_var = StringVar()
        ttk.Label(table_frame, text='Select the table to choose data from:').grid(column=0, row=0)
        for i, (table, ) in enumerate(self.connector.get_table_names()):
            rb = ttk.Radiobutton(
                    table_frame, 
                    text=table, 
                    variable=self.table_var, 
                    value=table, 
                    command=self.show_data
                    )
            rb.grid(column=0, row=i+1)
        table_frame.grid(column=0, row=0)

        choise_frame = ttk.Frame(self.root)
        self.x = StringVar()
        self.y = StringVar()
        self.x.set(str(None))
        self.y.set(str(None))
        
        self.x_table = StringVar()
        self.y_table = StringVar()
        self.x_table.set(str(None))
        self.y_table.set(str(None))
        
        ttk.Label(choise_frame, text="Use as data: ").grid(column=0, row=0)
        ttk.Label(choise_frame, text="Use as labels: ").grid(column=0, row=1)
        ttk.Label(choise_frame, textvariable=self.x).grid(column=1, row=0)
        ttk.Label(choise_frame, textvariable=self.y).grid(column=1, row=1)
        ttk.Button(choise_frame, text="Remove", command=lambda:self.x.set("None")).grid(column=2, row=0)
        ttk.Button(choise_frame, text="Remove", command=lambda:self.y.set("None")).grid(column=2, row=1)
        ttk.Button(choise_frame, text="Use selected values", command=self.select).grid(column=3, row=1)

        choise_frame.grid(column=1, row=1)
        
        self.tf = table_frame
        self.cf = choise_frame

    def select(self):
        #print(self.x.get(), self.y.get())
        self.root.destroy()

    def choose(self, value, var, table_var):
        var.set(value)
        table_var.set(self.table_var.get())

    def show_data(self):
        var = self.table_var.get()
        def handle_data_fields():
            
            columns, data_instances = self.connector.get_data("*", var, 10)
            
            if not hasattr(self, 'var_columns'):
                self.var_columns = []
                self.var_data = []
                self.data_buts = []
                self.label_buts = []
                initial = True
            else:
                initial = False

            for vc, column_name in enumerate(columns):
                if initial or len(self.var_columns) < vc+1:
                    #print(column_name, type(column_name))
                    self.var_columns.append(ttk.Label(self.vf, text=column_name))
                    
                    partial_x = partial(self.choose, column_name, self.x, self.x_table)
                    self.data_buts.append(
                            ttk.Button(
                                self.vf, 
                                text="As data", 
                                command=partial_x)
                            )
                    
                    partial_y = partial(self.choose, column_name, self.y, self.y_table)
                    self.label_buts.append(
                            ttk.Button(
                                self.vf, 
                                text="As labels", 
                                command=partial_y)
                            )
                else:
                    self.var_columns[vc].configure(text=column_name)
                    partial_x = partial(self.choose, column_name, self.x, self.x_table)
                    self.data_buts[vc].configure(command=partial_x)
                    partial_y = partial(self.choose, column_name, self.y, self.y_table)
                    self.label_buts[vc].configure(command=partial_y)

                self.var_columns[vc].grid(column=vc, row=1)
                self.data_buts[vc].grid(column=vc, row=2)
                self.label_buts[vc].grid(column=vc, row=3)
            
            
            datacount = 0
            for row, data_field in enumerate(data_instances):
                for column, data in enumerate(data_field):
                    if initial or datacount+1 > len(self.var_data):
                        self.var_data.append(ttk.Label(self.vf, text=data))
                    else:
                        self.var_data[datacount].configure(text=data)
                    self.var_data[datacount].grid(column=column, row=row+4)
                    datacount += 1
            
            #print(len(columns), len(self.var_columns))
            #print(datacount, len(self.var_data))

            for col in reversed(self.var_columns):
                #print(len(columns), len(self.var_columns))
                if len(columns) < len(self.var_columns):
                    col.grid_forget()
                    col.destroy()
                    self.data_buts[-1].grid_forget()
                    self.data_buts[-1].destroy()
                    self.label_buts[-1].grid_forget()
                    self.label_buts[-1].destroy()
                    del self.var_columns[-1]
                    del self.data_buts[-1]
                    del self.label_buts[-1]
                else:
                    break
            
            for drow in reversed(self.var_data):
                if datacount < len(self.var_data):
                    drow.grid_forget()
                    drow.destroy()
                    del self.var_data[-1]
                else:
                    break
            
        if not hasattr(self, 'vf'):
            variable_frame = ttk.Frame(self.root)
            self.variable_label = ttk.Label(variable_frame, text='Variables'+var)
            self.variable_label.grid(column=0, row=0)
            variable_frame.grid(column=1, row=0)
            self.vf = variable_frame
        else:
            self.variable_label.configure(text="Variable: "+var)

        handle_data_fields()
