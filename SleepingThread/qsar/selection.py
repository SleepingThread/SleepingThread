# -*- coding: utf-8 -*-

import os
import openbabel

import SleepingThread.qsar.qsardb as qdb

def createQSARPROJECTInput(input_filename,
        sel_filename,target_filename,output_filename="out.qp_out",
        countname="c1"):
    """
    input_filename [in] - filename of QSARPROJECT input filename
    sel_filename [in] - filename of .sdf selection
    target_filename [in] - filename with targets for selection
    """

    input_filename = os.path.abspath(input_filename)
    sel_filename = os.path.abspath(sel_filename)
    target_filename = os.path.abspath(target_filename)

    fout = open(input_filename,"w")
    fout.write("-selname "+str(sel_filename)+"\n")
    fout.write("-labelsname "+str(target_filename)+"\n")
    fout.write("-outfilename "+str(output_filename)+"\n")
    fout.write("-countname "+str(countname))
    fout.close()

    return

def splitSDF(sdf_filename,sel_folder,verbose=0):
    """
    """

    sdf_filename = os.path.abspath(sdf_filename)
    sel_folder = os.path.abspath(sel_folder)

    conv = openbabel.OBConversion()

    conv.SetInFormat("sdf")
    conv.SetOutFormat("mol")

    mol = openbabel.OBMol()
    if verbose>0:
        print "Process 1"
    sdf_not_empty = conv.ReadFile(mol,sdf_filename)

    _counter = 1
    while sdf_not_empty:
        mol_filename = sel_folder+"/"+str(_counter)+".mol"
        _counter += 1

        conv.WriteFile(mol,mol_filename)
        conv.CloseOutFile()

        if verbose>0:
            print "Process: "+str(_counter)

        sdf_not_empty = conv.Read(mol)

    return


def prepareSelectionQSARDB(qdb_folder,values_filename,sel_folder,steps=10000,
        verbose=0, with_hydrogens=True):
    """
    steps [in] - amount of steps for 3d structure optimization

    Build selection from QSARDB:
        1) Calculate mol 3d structures
        2) Create one .mol file per mol
        3) Create single .sdf file for whole selection
        4) Copy values file to sel_folder
        5) Create QSARDBPROJECT input file

        What to do after:
            Create files with surface data for mols
        
    """

    import shutil

    qdb_folder = os.path.abspath(qdb_folder)
    sel_folder = os.path.abspath(sel_folder)
    values_filename = os.path.abspath(values_filename)

    compounds = qdb.readCompounds(qdb_folder+"/compounds/compounds.xml")
    molecules = qdb.createMolDict(compounds,steps=steps,verbose=verbose,\
            with_hydrogens=with_hydrogens)
    qdb.writeSelection("one_mol_one_file",molecules,sel_folder)
    qdb.concatenateMols(sel_folder)
    shutil.copyfile(values_filename,sel_folder+"/values")

    # create input file for QSARPROJECT program
    createQSARPROJECTInput(sel_folder+"/selection.qp_input",
            sel_folder+"/selection.sdf",sel_folder+"/values")

    return

def prepareSelectionSDF(sdf_filename,values_filename,sel_folder,verbose=0):
    """
    """

    import shutil

    sdf_filename = os.path.abspath(sdf_filename)
    values_filename = os.path.abspath(values_filename)
    sel_folder = os.path.abspath(sel_folder)
 
    # split sdf_filename and write it to sel_folder
    splitSDF(sdf_filename,sel_folder,verbose=verbose)

    # copy sdf file and values file
    shutil.copyfile(sdf_filename,sel_folder+"/selection.sdf")
    shutil.copyfile(values_filename,sel_folder+"/values")

    # create input file for QSARPROJECT program
    createQSARPROJECTInput(sel_folder+"/selection.qp_input",
            sel_folder+"/selection.sdf",sel_folder+"/values")

    return
