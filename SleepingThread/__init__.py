# -*- coding: utf-8 -*-

import time
import cPickle as pickle
import zipfile
from zipfile import ZipFile
import os

class Timer(object):
    def __init__(self):
        self.start_time = time.time()
        return

    def start(self):
        self.start_time = time.time()
        return

    def __str__(self):
        # elapsed time 
        el_time = int(time.time() - self.start_time)

        res = str(self.hours(el_time))+":"+\
                str(self.minutes(el_time))+":"+\
                str(self.seconds(el_time))
        return res

    def hours(self,elapsed):
        return (elapsed//60)//60

    def minutes(self,elapsed):
        return (elapsed//60)%60

    def seconds(self,elapsed):
        return elapsed%60

"""
def ArgsGenerator(object):
    def __init__(self,sel_folder):
        self.sel_folder = sel_folder
        self.file_num = 1
        
    def next(self):
        if os.path.isfile(sel_folder+"/"+str(self.file_num)+".mol"):
            self.file_num += 1
            return sel_folder+"/"+str(self.file_num-1)+".mol"
        else:
            return None
"""

class Saver(object):
    def __init__(self,hours=0,mins=0,sec=None,n_iter=None,folder=None,generator=None,verbose=0,zip_output=False):
        """
        """
        
        self.zip_output = zip_output
        self.verbose = verbose

        if folder is None:
            raise Exception("You need explicitly specify folder name as folder=...")

        self.folder = folder
        # load from last file 
        file_num = 1
        while os.path.isfile(self.folder+"/"+str(file_num)):
            file_num += 1

        if file_num != 1:
            file_num -= 1
            # load generator
            _filename = self.folder+"/"+str(file_num)
            if not self.zip_output:
                _obj = pickle.load(open(_filename,"rb"))
            else:
                zf = ZipFile(_filename,"r")
                _pickle_str = zf.read(str(file_num))
                _obj = pickle.loads(_pickle_str)

            self.args_generator = _obj["generator"]
    
            file_num += 1
        else:
            self.args_generator = generator

        self.file_num = file_num

        self.hours = hours
        self.mins = mins
        self.sec = sec
        
        # number of seconds to save after
        self.n_sec = 0

        if sec is None and n_iter is None:
            n_iter = 1
            
        # number of iterations to save after
        self.n_iter = n_iter

        if sec is None:
            self.save_mode = "iter"
        else:
            self.save_mode = "time"
            self.n_sec = sec+mins*60+hours*60*60

        self.cur_time = time.time()
        self.el_iter = 0

        # data for args iter_generator
        self.args = None

        # object to save
        self.obj = None

        return

    def setArgsGenerator(self,generator):
        """
        """
        self.args_generator = generator
        return

    def saveObj(self,obj,protocol=2):
        """
        """

        if self.verbose > 0:
            print "Saving ... "

        self.obj = obj

        if not self.zip_output:
            fout = open(self.folder+"/"+str(self.file_num),"wb")
            # object to save
            sobj = {"obj":obj,"generator":self.args_generator}
            pickle.dump(sobj,fout,protocol)
            fout.close()
        else:
            # object to save
            sobj = {"obj":obj,"generator":self.args_generator}
            _pickle_str = pickle.dumps(sobj,protocol)

            zf = ZipFile(self.folder+"/"+str(self.file_num),"w",\
                    compression=zipfile.ZIP_DEFLATED)
            zf.writestr(str(self.file_num),_pickle_str)
            zf.close()

        self.file_num += 1

        return

    def restoreObj(self,obj):
        """
        """
        del obj[:]

        # read all files in self.folder
        cur_file_num = 1
        while os.path.isfile(self.folder+"/"+str(cur_file_num)):
            cur_filename = self.folder+"/"+str(cur_file_num)

            if not self.zip_output:
                cur_obj = pickle.load(open(cur_filename,"rb"))
            else:
                zf = ZipFile(cur_filename,"r")
                _pickle_str = zf.read(str(cur_file_num))
                cur_obj = pickle.loads(_pickle_str)
            
            cur_file_num += 1
            obj.append(cur_obj["obj"])

        return

    def next(self,obj):
        """
        !!! Now object can be list only
        save object if it needs
        and return next iteration
        """
        save_obj = False
        if self.save_mode == "iter":
            if self.el_iter >= self.n_iter:
                self.el_iter = 0
                save_obj = True

            self.el_iter += 1

        elif self.save_mode == "time":
            _now_time = time.time()
            if _now_time-self.cur_time>=self.n_sec:
                self.cur_time = _now_time
                save_obj = True

        else:
            raise Exception("No such save mode")

        if save_obj:
            self.saveObj(obj)

            #empty obj
            del obj[:]

        self.args = self.args_generator.next()

        if self.args is not None:
            return True
        else:
            self.saveObj(obj)

            # full restore object
            self.restoreObj(obj)
            return False

        return 

    def getArgs(self):
        return self.args


