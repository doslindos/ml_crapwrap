from importlib.util import find_spec
from GUI.GUI_functions import Mysql_data_selector, define_relation
import mysql.connector

class MySQL_Connector:
    # Handles mysql.connector calls

    def __init__(self):
        # Initializes the connector
        if find_spec('credentials'):
            from credentials import MySQL_connector_params
            self.db = mysql.connector.connect(**MySQL_connector_params)
        else:
            print("Define Mysql credentials... Check guide MySQL credentials!")
            exit()

    def close_connection(self):
        # Closes the connection
        self.db.close()
        print("Database connection closed...")
    
    def get_table_names(self):
        # Gets tables from a chocen dataset
        cursor = self.db.cursor()
        cursor.execute("SHOW TABLES")
        return cursor

    def get_data(self, select_q, from_q, inst=None, as_dict=False):
        # Makes a dataset query
        cursor = self.db.cursor(dictionary=as_dict, buffered=True)
        sql = "SELECT "+select_q+" FROM "+from_q
        #print(sql)
        cursor.execute(sql)
    
        if not as_dict:
            columns = [col_name for col_name in cursor.column_names]
            if inst is None:
                data = [d for d in cursor]
            else:
                data = [d for d in cursor.fetchmany(inst)]
        
            cursor.close()
        
            return (columns, data)
        else:
            data = [d for d in cursor]
            return data

    def select_data(self):
        #
        selector = Mysql_data_selector(self)
        data_table = selector.x_table.get()
        data_name = selector.x.get()
        label_table = selector.y_table.get()
        labels_name = selector.y.get()
        print("Selected: ", data_name, labels_name)
        if data_name == 'None':
            print("Data was not selected...")
            exit()
        else:
            # Define relation
            if data_table != label_table:
                print("Data and labels are from different tables!")
                data_cols, _ = self.get_data("*", data_table, 1)
                label_cols, _ = self.get_data("*", label_table, 1)
                dr = define_relation(data_cols, label_cols)
                relations = (dr.relation_1.get(), dr.relation_2.get())
                #print(relations)
                data = self.get_data(
                        "A."+relations[0]+", A."+data_name+" as data, max(B."+labels_name+") as label", 
                        data_table+" A join "+label_table+" B on (A."+relations[0]+"=B."+relations[1]+") GROUP BY A."+relations[0],
                        as_dict=True
                        )

            else:
                if labels_name != 'None':
                    q = data_name+label_name
                    print("No labels choosed...")
                else:
                    q = data_name

                data = self.get_data(data_table, q, as_dict=True)
            
            return data
