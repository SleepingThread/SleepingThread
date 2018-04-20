# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# openbabel imports
import openbabel

def readCompounds(filename):
    """
    Read compounds from qsardb .xml file
    filename [in]: filename with .xml file with compounds
    return: dictionary, key = <id in xml>, value = <InChI in xml>
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    res = {}
    for elem in root:
        c_id = int(elem.find("{http://www.qsardb.org/QDB}Id").text)
        c_inchi = elem.find("{http://www.qsardb.org/QDB}InChI").text
        
        res[c_id] = c_inchi

    return res

def createMolDict(mol_dict,steps=500,verbose=0):
    """
    Creates 3d coordinates from InChI using openbabel mmff94
    mol_dict [in] - dict where values - InChI strings
    steps [in] - steps for openbabel optimizer
    return: dictionary, key = <key from mol_dict>, value = <openbabel.OBMol>
    """
    #mol_dict[<mol_id>]=<InChI string>
    
    conv = openbabel.OBConversion()
    conv.SetInFormat("inchi")
    #conv.SetInAndOutFormats("inchi","mol")
   
    #create builder
    builder = openbabel.OBBuilder()

    #find Force Field
    ff = openbabel.OBForceField.FindForceField("mmff94")
    if ff is None:
        raise Exception("Cannot find Force Field mmff94")
        return None

    new_dict = {}

    for key in mol_dict:
        if verbose>0:
            sys.stdout.write("\rProcess: "+str(key))
            sys.stdout.flush()

        mol = openbabel.OBMol()
        conv.ReadString(mol,mol_dict[key])

        #create 3D
        builder.Build(mol)

        mol.AddHydrogens()

        ff.Setup(mol)
        ff.SteepestDescent(steps)
        ff.GetCoordinates(mol)

        new_dict[key] = mol

    #conv.SetOutFormat("mol")
    #conv.WriteFile(mol,"test.mol")

    if verbose>0:
        print " "

    return new_dict

def writeSelection(sel_type,mol_dict,path):
    """
    Create and save files from dictionary of molecules
    mol_dict [in] - dictionary, values = <openbabel.OBMol>
    sel_type [in] - 
        1) one_mol_one_file - 
            create mol file with name <path>/<key>.mol
            for each molecule
    path [in] - Specify <path> for molecule files

    return: nothing
    """
    if sel_type=="one_mol_one_file":
        #filename = <id>.mol

        #create path if not exists
        if not os.path.exists(path):
            os.makedirs(path)
        elif not os.path.isdir(path):
            raise Exception(path+" exists and it is not folder")
            return

        conv = openbabel.OBConversion()
        conv.SetOutFormat("mol")

        for key in mol_dict:
            conv.WriteFile(mol_dict[key],path+"/"+str(key)+".mol")

    return

def concatenateMols(molpath,maxid=None,sel_filename="selection.sdf"):
    """
    functions concatenates mol files molpath+"/"+<ind>.mol into  
        sdf file molpath+"/"+sel_filename
    molpath [in] - path for reading molecules from it
    maxid [in] - specify last name molpath+"/"+<maxid>.mol to read
    filename [in] - specify created .sdf filename
    return : nothing
    """
    conv = openbabel.OBConversion()
    conv.SetInFormat("mol")
    conv.SetOutFormat("sdf")

    sdf_path = molpath+"/"+sel_filename

    cur_ind = 1
    curfilename = molpath+"/"+str(cur_ind)+".mol"
    cur_exists = False

    if os.path.isfile(curfilename):
        mol = openbabel.OBMol()
        cur_exists = conv.ReadFile(mol,curfilename)

        conv.WriteFile(mol,sdf_path)
    else:
        #There are no 1.mol in molpath
        return

    print "Process file: "+curfilename

    while cur_exists:
        cur_ind += 1
        curfilename = molpath+"/"+str(cur_ind)+".mol"
        if not os.path.isfile(curfilename):
            break
        if maxid is not None:
            if cur_ind>maxind:
                break
        print "Process file: "+curfilename
        mol = openbabel.OBMol()
        conv.ReadFile(mol,curfilename)

        conv.Write(mol)

    conv.CloseOutFile() 

    return

def readDescriptors(path,folder_list,col_name="from_csv"):
    """
    path [in]: path of folders 
    folder_list [in]: descriptors folder names
    col_name [in]: take column name from csv file ("from_csv") or
        from folder_list ("from_folder_list")
    return: pandas.DataFrame
    """
    
    if col_name == "from_csv":
        col_list = []
        for foldername in folder_list:
            fn = path+"/"+foldername+"/values"
            feature = pd.read_csv(fn,delimiter="\t")
            col_list.append(feature[feature.columns[-1]])

        table = pd.DataFrame(col_list).transpose()
    elif col_name == "from_folder_list":
        col_list = {}
        for foldername in folder_list:
            fn = path+"/"+foldername+"/values"
            feature = pd.read_csv(fn,delimiter="\t")
            col_list[foldername] = feature[feature.columns[-1]].values

        table = pd.DataFrame(col_list)

    table = table.astype(np.float32)

    return table

