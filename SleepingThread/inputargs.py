# -*- coding: utf-8 -*-
"""
You need to create function:
readConsoleInput(argv,argdict):
    err = []
    res = readInputInstruction(argv,"--help","none",argdict,err)
    if res==1:
        #print something

    if res==0:
        #set default parameter

    if len(err)>0:
        return -1

    return 0

Or you need to create input_tmpl:
    input_tmpl={
        "varname":(
            "defaultvalue",
            "description"
        ),
        ...
    }

Or you need to create input_tmpl as list:
    input_tmpl = [
        ("help",(
            False,
            "head help description"
        )),
        ("varname",(
            "default_value",
            "description"
        )), 
        ...
    ]

    input_arguments = readInputArguments(input_tmpl)

    Use cmd options as: --varname <varvalue>

"""

import sys

def cmdisinarray(iname,argv):
    """
    Function check if iname in argv list
    """
    ind = 0
    pos = -1
    while ind<len(argv):
        if argv[ind]=="--":
            #do not check next element in argv
            ind+=1
        elif argv[ind]==iname:
            pos = ind
            break

        ind+=1

    return pos

def readInputInstruction(argv,iname,ftype,argdict,err=[]):
    """
    iname - instruction name
    ftype - type of field
    argdict - dictionary of input instructions
        keys - input instruction names
    err - list of error messages
    """
    pos = cmdisinarray(iname,argv)
    if pos!=-1:
        if ftype=="bool":
            argdict[iname]=True
        elif ftype!="none":
            #check length
            if pos+1>=len(argv):
                err.append("readInputInstruction: Input "+iname+\
                        " must have "+ftype+"parameter")
                return -1
            if argv[pos+1][:2]=="--":
                if len(argv[pos+1])!=2:
                    err.append("readInputInstruction: Input "+iname+\
                            " parameter can't begin with --")
                    return -1
                else:
                    pos += 1
            #check length
            if pos+1>=len(argv):
                err.append("readInputInstruction: Input "+iname+\
                        " must have "+ftype+"parameter")
                return -1
            if ftype=="int":
                argdict[iname]=int(argv[pos+1])
            elif ftype=="double":
                argdict[iname]=float(argv[pos+1])
            elif ftype=="str":
                argdict[iname]=argv[pos+1]

        return 1
    else:
        if ftype=="bool":
            argdict[iname]=False
        return 0

    return 0

def _gettype(val):
    if type(val)==int:
        return 'int'
    elif type(val)==float:
        return 'double'
    elif type(val)==str:
        return 'str'
    elif type(val)==bool:
        return 'bool'

    return 'none'

def _print_help(input_tmpl,input_tmpl_list):
    if "help" in input_tmpl:
        print input_tmpl['help'][1]
    print "AUTOHELP:"
    if input_tmpl_list is not None:
        #print help from list
        for el in input_tmpl_list:
            if el[0]=="help":
                continue
            print "--"+el[0]+" == "+repr(el[1][0])
            for line in el[1][1].split("\n"):
                print "  "+line
    else:
        #print unsorted help from dict
    	for key in input_tmpl:
    		if key=="help":
    			continue
    		print "--"+key+" == "+repr(input_tmpl[key][0])
    		for line in input_tmpl[key][1].split("\n"):
    			print "  "+line
    return

def readInputArguments(input_tmpl,args=None):
    """
    input_parameters - in and out:
        'varname':(defaultval,description)
    """
    input_tmpl_list = None
    if isinstance(input_tmpl,list):
        listinput = True
        input_tmpl_list = input_tmpl
        input_tmpl = {}
        for el in input_tmpl_list:
            input_tmpl[el[0]] = el[1]

    if not args:
        args = sys.argv
    input_arguments = {}
    err = []
    if cmdisinarray("--help",args)!=-1:
        _print_help(input_tmpl,input_tmpl_list)
        input_arguments["--help"] = True
    
    for key in input_tmpl:
        if key=="help":
            continue
        _type = _gettype(input_tmpl[key][0])
        if _type=='none':
            print "key: "+key+" - wrong type" 
            _print_help(input_tmpl,input_tmpl_list)
            return None
        res = readInputInstruction(args,"--"+key,_gettype(input_tmpl[key][0]),input_arguments,err)
        if res==0:
            input_arguments["--"+key] = input_tmpl[key][0]

        if len(err)>0:
            print "ERROR:"
            for el in err:
                print "  "+el

            _print_help(input_tmpl,input_tmpl_list)
            return None

    return input_arguments

