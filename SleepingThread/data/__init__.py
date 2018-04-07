# -*- coding: utf-8 -*-

import math

# pandas imports
import pandas as pd

def removeNAN(oldtable):
    """
    oldtable [in] - pandas DataFrame or list<list<float>>
    return: pandas table without NAN columns
    """
    if isinstance(oldtable,list):
        oldtable = pd.DataFrame(oldtable)

    not_nan_col_names = []
    for col_name in oldtable.columns:
        #get Series
        ser = oldtable[col_name]
        _not_nan = True
        for el in ser:
            if math.isnan(el):
                _not_nan = False
                break
        if _not_nan:
            not_nan_col_names.append(col_name)

    newtable = oldtable[not_nan_col_names]

    return newtable

def concatTables(pandas_tables):
    """
    Concatenate pandas.DataFrames with descriptors, 
        check if there are columns with the same names
    """

    col_names = set()
    for table in pandas_tables:
        for col_name in table.columns:
            if col_name in col_names:
                raise Exception("Two or more columns with the same name: "+str(col_name))
            col_names.add(col_name)


    result = pd.concat(pandas_tables,axis=1)

    return result

