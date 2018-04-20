# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin, clone
from SleepingThread.qsar import descriptors as descr
from SleepingThread.st_gmdh import GMDH

import SleepingThread.surface as surface
import plotly.offline as po
import plotly.graph_objs as go
from SleepingThread.qsar import graphics

def normalize(vec):
    vec /= (np.sum(vec**2))**0.5
    return vec

class QSARModel1(BaseEstimator, RegressorMixin):
    def __init__(self,estimator=GMDH(),n_image_clusters=10,n_prop_clusters=4,
                verbose=1):
        self.estimator = estimator
        self.n_image_clusters = n_image_clusters
        self.n_prop_clusters = n_prop_clusters
        self.verbose = verbose
        
        self.sptype = None
        self.trainer = None
        return
       
    def getSPList(self,X):
        """
        X - SPGeneratorData
        X[i] = {"id":0,"sp":[[image, ... ],...],"descriptors":[<row of features>]}
        """

        # get sp_list
        sp_list = []
        for el in X:
            sp_list.append(el["sp"])        
        
        return sp_list
        
    def fit(self,X,y):
        """
        X - SPGeneratorData
        X[i] = {"id":0,"sp":[[image, ... ],...],"descriptors":[<row of features>]}
        """
        sp_list = self.getSPList(X)

        self.sptype = descr.SPType(n_image_clusters=self.n_image_clusters,n_prop_clusters=self.n_prop_clusters)
        self.sptype.fit(sp_list)
        mat = descr.createDescriptors(sp_list,self.sptype)
       
        if "descriptors" in X[0]:
            # build features_mat
            features_list = []
            for el in X:
                features_list.append(el["descriptors"])

            features_mat = np.array(features_list)

            # concatenate mat and features_mat
            mat = np.concatenate([mat,features_mat],axis=1)

        self.trainer = clone(self.estimator)
        self.trainer.fit(mat,y)
        return
    
    def predict(self,X):
        """
        X - SPGeneratorData
        X[i] = {"id":0,"sp":[[images, ... ],...],"descriptors":[<row of features>]}
        """
        sp_list = self.getSPList(X)

        mat = descr.createDescriptors(sp_list,self.sptype)

        if "descriptors" in X[0]:
            # build features_mat
            features_list = []
            for el in X:
                features_list.append(el["descriptors"])

            features_mat = np.array(features_list)

            # concatenate mat and features_mat
            mat = np.concatenate([mat,features_mat],axis=1)

        return self.trainer.predict(mat)

    def getImageClusterProp(self,X):
        """
        X[i] ={"id":0,"sp":[[image,...],...],"descriptors":[<row of features>],
            "labels":[<segment labels>]}
        """
        sp_list = self.getSPList(X)

        # segm_im_types[<mol_ind>][<mol_segment_number>] = <segment_type>
        segm_im_types = self.sptype.getImageType(sp_list)

        image_cluster_prop_list = []
        for mol_ind in xrange(len(X)):
            mol_image_cluster_prop = []
            image_cluster_prop_list.append(mol_image_cluster_prop)

            # mol_segm_im_types[<segment label>] = <image type>
            mol_segm_im_types = segm_im_types[mol_ind]

            # mol_segm - list of segment labels for surface points
            mol_segm = X[mol_ind]["labels"]

            # for surface points get image type by segment label
            for segm_num in mol_segm:
                mol_image_cluster_prop.append(mol_segm_im_types[segm_num])

        return np.asarray(image_cluster_prop_list)
    
    def getSegmentationProp(self,X):
        """
        X[i] ={"id":0,"sp":[[image,...],...],"descriptors":[<row of features>],
            "labels":[<segment labels>]}
        """       
        segm_labels_list = []
        for el in X:
            segm_labels_list.append(el["labels"])

        return np.asarray(segm_labels_list)
    
    def getSurfaceProp(self,X):
        """
        Return [[<props for mol surface points>] ... ], where prop values - for example MMFF94
        X[i] ={"id":0,"sp":[[image,...],...],"descriptors":[<row of features>],
            "labels":[<segment labels>],"props":[<MMFF94 props>]}
        """       
        surface_prop_list = []
        for el in X:
            surface_prop_list.append(el["props"])

        return np.asarray(surface_prop_list)
        
    def _change_CS_1(self,points,center,normal):
        """
        Output:
            points, center, normal
        """
        
        # centrate points
        points = points-center
        
        # select axis: [1,0,0] or [0,1,0]:
        if normal[0]>0.75:
            # select [0,1,0]
            x_axis = normal
            y_axis = normalize(np.array([0.0,1.0,0.0])-normal[1]*normal)
        else:
            # select [1,0,0]
            x_axis = normal
            y_axis = normalize(np.array([1.0,0.0,0.0])-normal[0]*normal)
            
        z_axis = normalize(np.cross(x_axis,y_axis))
        
        rot = np.array([x_axis,y_axis,z_axis])
        rot = rot.transpose()
        
        points = np.dot(points,rot)
        
        return points,np.array([0.0,0.0,0.0]),np.array([1.0,0.0,0.0])
        
    def drawSegments(self,X,image_label=None,single_label=None,start=-1,end=-1,singlefigure=True,drawprop=False):
        """
        To draw segment with selected property (image_label or single_label)
        Segment can be drawed with property (??)
            add start,end
            ability to draw center, normal
            ability to draw 
            
            how to rotate for identical normals
            draw on one or to different figures
        """
        
        if image_label is None and single_label is None:
            raise Exception("image_label or single_label must be not None")
        
        sp_list = self.getSPList(X)
        
        # segm_types[<mol_ind>][<mol_segment_number>] = <segment_type>
        if image_label is not None:
            label = image_label
            segm_types = self.sptype.getImageType(sp_list)
        else:
            label = single_label
            segm_types = self.sptype.predict(sp_list)
        
        if drawprop:
            surf_prop_list = self.getSurfaceProp(X)
            segm_prop_list = self.getSegmentationProp(X)
            segm_surf_prop_list = []
        else:
            segm_surf_prop_list = None
        
        points_list = []
        mesh_index_list = []
        centers_list = []
        normals_list = []
        descriptions_list = []
        for mol_ind,mol_sp in enumerate(sp_list):
            for sp_ind,sp in enumerate(mol_sp):
                if segm_types[mol_ind][sp_ind]==label:
                    points,mesh_index = sp[3]
                    center,normal = sp[4]
                    
                    # modify CS so normal = [1,0,0]
                    points,center,normal = self._change_CS_1(points,center,normal)
                    
                    points_list.append(points)
                    mesh_index_list.append(mesh_index)
                    centers_list.append(center)
                    normals_list.append(normal)
                    descriptions_list.append(str((mol_ind,sp_ind)))
                    
                    if drawprop:
                        segm_surf_prop_list.append(np.array(surf_prop_list[mol_ind][segm_prop_list[mol_ind]==sp_ind]))
                  
        
        draw_res = graphics.drawSurfaces(points_list,mesh_index_list,segm_surf_prop_list,
                                         start=start,end=end,draw=False,
                                         singlefigure=singlefigure, descriptions_list=descriptions_list)
        
        po.iplot(draw_res[2])
               
        # return min and max props
        return draw_res[0],draw_res[1]
    
    def drawImages(self,X,image_label=None,single_label=None,start=-1,end=-1):
        """
        """
        if image_label is None and single_label is None:
            raise Exception("image_label or single_label must be not None")
        
        sp_list = self.getSPList(X)
        
        # segm_types[<mol_ind>][<mol_segment_number>] = <segment_type>
        if image_label is not None:
            label = image_label
            segm_types = self.sptype.getImageType(sp_list)
        else:
            label = single_label
            segm_types = self.sptype.predict(sp_list)
        
        images_list = []
        points_list = []
        mesh_index_list = []
        centers_list = []
        normals_list = []
        descriptions_list = []
        for mol_ind,mol_sp in enumerate(sp_list):
            for sp_ind,sp in enumerate(mol_sp):
                if segm_types[mol_ind][sp_ind]==label:
                    images_list.append(sp[0])
                    descriptions_list.append(str((mol_ind,sp_ind)))

        graphics.drawImages(images_list,descriptions_list=descriptions_list,start=start,end=end)
        
        return
   

class QSARModel(BaseEstimator, RegressorMixin):
    def __init__(self,sp_generator_fun,estimator=GMDH(),n_sp_clusters=10,n_image_clusters=10,n_prop_clusters=4,
                imsize=7,verbose=1):
        self.sp_generator_fun = sp_generator_fun
        self.estimator = estimator
        self.n_sp_clusters = n_sp_clusters
        self.n_image_clusters = n_image_clusters
        self.n_prop_clusters = n_prop_clusters
        self.imsize = imsize
        self.verbose = verbose
        
        self.sptype = None
        self.trainer = None
        return
        
    def fit(self,X,y):
        """
        X - SPGeneratorData
        X[i] = {"id":0,"data":[mol_points, mol_props],"features":[<row of features>]}
        """
        sp_list, segm_list = self.sp_generator_fun(X,self.n_sp_clusters,self.imsize) 

        self.sptype = descr.SPType(n_image_clusters=self.n_image_clusters,n_prop_clusters=self.n_prop_clusters)
        self.sptype.fit(sp_list)
        mat = descr.createDescriptors(sp_list,self.sptype)
       
        if "features" in X[0]:
            # build features_mat
            features_list = []
            for el in X:
                features_list.append(el["features"])

            features_mat = np.array(features_list)

            # concatenate mat and features_mat
            mat = np.concatenate([mat,features_mat],axis=1)

        self.trainer = clone(self.estimator)
        self.trainer.fit(mat,y)
        return
    
    def predict(self,X):
        """
        X - SPGeneratorData
        X[i] = {"id":0,"data":[mol_points, mol_props],"features":[<row of features>]}
        """
       
        # X[i] <-> sp_list[i]
        sp_list, segm_list = self.sp_generator_fun(X,self.n_sp_clusters,self.imsize)
        
        mat = descr.createDescriptors(sp_list,self.sptype)

        if "features" in X[0]:
            # build features_mat
            features_list = []
            for el in X:
                features_list.append(el["features"])

            features_mat = np.array(features_list)

            # concatenate mat and features_mat
            mat = np.concatenate([mat,features_mat],axis=1)

        return self.trainer.predict(mat)

    def createImageSegmentationProp(self,X):
        """
        X - SPGeneratorData
        X[i] = {"id":0,"data":[mol_points, mol_props],"features":[<row of features>]}
        """
        
        sp_list, segm_list = self.sp_generator_fun(X,self.n_sp_clusters,self.imsize)
        
        return segm_list

    def createImageClusterProp(self,X):
        """
        X[i] ={"id":0,"data":[mol_points, mol_props],"features":[<row of features>]}
        """

        sp_list, segm_list = self.sp_generator_fun(X,self.n_sp_clusters,self.imsize)

        # segm_im_types[<mol_ind>][<mol_segment_number>] = <segment_type>
        segm_im_types = self.sptype.getImageType(sp_list)

        image_cluster_prop_list = []
        for mol_ind in xrange(len(X)):
            mol_image_cluster_prop = []
            image_cluster_prop_list.append(mol_image_cluster_prop)

            # mol_segm_im_types[<segment label>] = <image type>
            mol_segm_im_types = segm_im_types[mol_ind]

            # mol_segm - list of segment labels for surface points
            mol_segm = segm_list[mol_ind]

            # for surface points get image type by segment label
            for segm_num in mol_segm:
                mol_image_cluster_prop.append(mol_segm_im_types[segm_num])

        return image_cluster_prop_list,segm_list
    
    def getSPImageClusterLabels(self,X):
        """
        X[i] ={"id":0,"data":[mol_points, mol_props],"features":[<row of features>]}
        """

        sp_list, segm_list = self.sp_generator_fun(X,self.n_sp_clusters,self.imsize)

        # segm_im_types[<mol_ind>][<mol_segment_number>] = <segment_type>
        segm_im_types = self.sptype.getImageType(sp_list)
        

        return np.asarray(sp_list),segm_im_types
