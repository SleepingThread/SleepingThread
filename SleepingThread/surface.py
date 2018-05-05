# -*- coding: utf-8 -*-

"""
"""

import re
import numpy as np

class Surface(object):

    @staticmethod
    def read_mesh_index(filename):
        mesh_index = []
        fin = open(filename,"r")
        for line in fin:
            line = line.strip()
            arr = re.split(",",line)
            if len(arr)<3:
                break
            mesh_index.append(int(arr[0]))
            mesh_index.append(int(arr[1]))
            mesh_index.append(int(arr[2]))
    
        fin.close()

        return mesh_index

    @staticmethod
    def read_points(filename):
        points = []
        fin = open(filename,"r")
        for line in fin:
            line = line.strip()
            arr = re.split("\s+|,",line)
            if len(arr)<3:
                break
            points.append(np.array([float(arr[0]),float(arr[1]),float(arr[2])]))
    
        fin.close()
    
        return points

    @staticmethod
    def readSurface(points_filename,mesh_index_filename):
        points = Surface.read_points(points_filename)
        mesh_index = Surface.read_mesh_index(mesh_index_filename)

        return Surface(points,mesh_index)

    @staticmethod
    def calculateSurfaceCenter(points):
        """
        """

        minind = -1
        minval = float("+inf")

        for ind,el in enumerate(points):
            _aver = np.average(np.linalg.norm(points-el,axis=1))
            if _aver<minval:
                minval = _aver
                minind = ind

        return points[minind]

    @staticmethod
    def calculateNormal(points,mesh_index,center,percent=0.8):
        """
        """

        surf = Surface(points,mesh_index)
        ss_points,ss_mesh_index = surf.getSubSurface(surf.getSubset(center,percent=percent))
        if len(ss_mesh_index)==0:
            ss_points,ss_mesh_index = \
                    surf.getSubSurface(surf.getSubset(center,percent=2.0))

        normal = np.array([0.0,0.0,0.0])
        for el in ss_mesh_index:
            a = ss_points[el[1]]-ss_points[el[0]]
            b = ss_points[el[2]]-ss_points[el[0]]
            c = (ss_points[el[0]]+ss_points[el[1]]+ss_points[el[2]])/3.0

            normal += 1.0/(1.0+np.sum((c-center)**2)) * np.cross(a,b)

        return normal/np.linalg.norm(normal)

    def __init__(self,points,mesh_index):
        self.points = np.array(points)
        self.mesh_index = np.array(mesh_index).reshape(-1,3)

        self.labels = None

        self.segments = None
        self.segm_props = None

        return

    def getSubset(self,center,radius=None,percent=0.2,output="boolmask"):
        """
        output = "index" | "indexes" | "boolmask" | "point" | "points"
        """
        if radius is None:
            surf_center = np.average(self.points,axis=0) 
            # max deviation
            max_dev = np.max(np.linalg.norm(self.points-surf_center,axis=1))
            # radius
            radius = percent*max_dev

        if output=="index" or output=="indexes" or output=="boolmask":
            subset = np.zeros([len(self.points)],dtype=np.bool)
            for ind,el in enumerate(self.points):
                if np.linalg.norm(el-center)<radius:
                    subset[ind] = True
        elif output=="point" or output=="points":
            subset = []
            for ind,el in enumerate(self.points):
                if np.linalg.norm(el-center)<radius:
                    subset.append(el)
        else:
            raise Exception("Wrong output argument value")

        return subset

    def getSubSurface(self,bool_mask):
        """
        """

        # subsurface points has it's own indices, different from 
        # this indices inside surface
        new_inds = []
        _counter = 0
        for i in xrange(len(bool_mask)):
            if bool_mask[i]:
                new_inds.append(_counter)
                _counter += 1
            else:
                new_inds.append(-1)

        new_inds = np.array(new_inds)

        segm_mesh_index = []
        for el in self.mesh_index:
            if np.sum(bool_mask[el])==3:
                segm_mesh_index.append(new_inds[el])

        segm_points = self.points[bool_mask]

        return segm_points, np.array(segm_mesh_index)

    def getSegment(self,label):
        """
        """

        if self.labels is None:
            raise Exception("You need to call setLabels or setSegmentation")

        bool_mask = self.labels==label

        return self.getSubSurface(bool_mask)

    def createSegments(self,labels):
        """
        return 
            list<[ list<point>,<mesh indexes>]>,
            list<[ center , normal ]>
        """
        
        if len(labels) != len(self.points):
            raise Exception("labels has wrong dimensions")

        self.labels = labels

        self.segments = []
        self.segm_props = []

        # get number of segments
        n_segments = np.max(labels)+1

        for segm_label in xrange(n_segments):
            # get segment subsurface
            segm_points,segm_mesh_index = self.getSegment(segm_label)
            # get segment properties
            segm_center = Surface.calculateSurfaceCenter(segm_points)
            segm_normal = Surface.calculateNormal(segm_points,segm_mesh_index,segm_center)

            self.segments.append([segm_points,segm_mesh_index])
            self.segm_props.append([segm_center,segm_normal])

        return self.segments, self.segm_props
