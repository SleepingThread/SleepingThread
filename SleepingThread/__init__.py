# -*- coding: utf-8 -*-

import time

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


