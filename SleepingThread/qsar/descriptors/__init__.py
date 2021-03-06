# -*- coding: utf-8 -*-
"""

Examples of using pybel:

import pybel
mol = pybel.readstring("smi","CC(=O)Br")
mol.make3D()

#create files from molecule
print(mol.write("sdf"))
print(mol.write("mol"))
print(mol.write("pdb"))

#read molecule from file
mol2 = pybel.readfile("mol","./data/test.mol").next()

OBGridData* GetGrid()

Example of using openbabel:

import openbabel

def getAtomTypes(mol,typename):
    typelist = []
    ttab = openbabel.ttab
    ttab.SetFromType("INT")
    if not ttab.SetToType(typename):
        print "NO SUCH typename: ",typename
    atoms = openbabel.OBMolAtomIter(mol)
    for a in atoms:
        src = a.GetType()
        dst = ttab.Translate(src)
        typelist.append(dst)

    return typelist

def readMol(filename):
    conv = openbabel.OBConversion()
    mol = openbabel.OBMol()

    res = conv.ReadFile(mol,filename)
    if res is None:
        return None
    else:
        return mol

    return mol

def makeForceField(mol):
    ff = openbabel.OBForceField.FindForceField("mmff94")
    ff.Setup(mol)
    return ff

Code examples for openbabel:
    mol = openbabel.OBMol()
    res = conv.ReadFile(mol,"./test.mol")
    #return molar mass
    mol.GetMolWt()
    mol.NumAtoms()
    atoms = openbabel.OBMolAtomIter(mol)
    ttab = openbabel.ttab
    ttab.SetFromType("INT")
    ttab.SetToType("MM2")
    for a in atoms:
        src = a.GetType()
        print src, a.GetPartialCharge()
        dst = ttab.Translate(src)
        print "MM2: ",dst

    ff = openbabel.OBForceField.FindForceField("mmff94")
    ff.Setup(mol)
    ff.E_VDW()

    a.SetAtomicNum(6)

"""

"""
@todo Add ver.2 of main axis for segment 
    Add description for Special points
    Add normal calculation
    Add function for working with surfaces, subsurfaces
"""


import re
import numpy as np
from subprocess import check_call

# sklearn imports
from sklearn import decomposition

# CDK imports in function createCDKDescriptors 

# openbabel imports
import openbabel

# local imports
import SleepingThread.qsar as qsar
import SleepingThread.metrics as metrics

def read_mol(molfilename):
    conv = openbabel.OBConversion()
    conv.SetInFormat("mol")
    mol = openbabel.OBMol()
    res = conv.ReadFile(mol,molfilename)
    if res is None:
        return None
    else:
        return mol

    #while(res):
    #   print mol.GetMolWt()
    #   mol.Clear()
    #   conv.Read(mol)
    #
    #print mol.GetMolWt()
    #conv.Read(mol2) #if there are no one molecule

    return mol

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

def calculateTrSquare(p1,p2,p3):
    tmp = p2-p1
    a = np.linalg.norm(tmp)
    tmp = p3-p1
    b = np.linalg.norm(tmp)
    tmp = p3-p2
    c = np.linalg.norm(tmp)

    perim = (a+b+c)/2.0

    return (perim*(perim-a)*(perim-b)*(perim-c))**0.5

#Weighted mesh
def createWCloud(mesh_index,points):
    wcloud = []
    weights = []
    #walk through all triangles
    triangles = len(mesh_index)/3
    for i in xrange(triangles):
        pind1 = mesh_index[i*3]
        pind2 = mesh_index[i*3+1]
        pind3 = mesh_index[i*3+2]

        p1 = points[pind1]
        p2 = points[pind2]
        p3 = points[pind3]

        square = calculateTrSquare(p1,p2,p3)

        #new point
        new_point = 1.0/3.0*(points[pind1]+points[pind2]+points[pind3])

        wcloud.append(new_point)
        weights.append(square)
        
    return wcloud, weights 

def centrateData(points,weights):
    center = np.array([0.0,0.0,0.0])
    counter = 0
    for i in xrange(len(points)):
        center += points[i]*weights[i]
        counter += weights[i]

    center = 1.0/counter * center;

    for i in xrange(len(points)):
        points[i] -= center

    return points,center

def calculateMMFF94(points,mol_file,ffname="mmff94",
        prop_type="El",prob_type="1",prob_charge=1):
    """
    prop_type [in] = El | ElVDW | VDW
    """
    #calculate MMFF94 values in points
    mol = read_mol(mol_file)
    ff = openbabel.OBForceField.FindForceField(ffname)
    if ff is None:
        print "No such forcefield ",ffname
        return None

    prop = []
    ff.Setup(mol)

    for point in points:
        #what calculated, probe atom type, partial charge, probe atom coordinates
        prop.append(ff.calculateEl_VDW(prop_type,\
                prob_type,prob_charge,point[0],point[1],point[2]))

    return prop

def calculateCloudMMFF94(prop,mesh_index):
    #create prop for each triangle
    tr_prop = []
    mesh_size = len(mesh_index)/3
    for i in xrange(mesh_size):
        val = (prop[mesh_index[3*i]]+prop[mesh_index[3*i+1]]+prop[mesh_index[3*i+2]])/3.0
        tr_prop.append(val)

    return tr_prop

def buildImage(cloud,weights,prop,normal,imsize):
    """
    Need to specify image size 
    """
    #matrix with components rgb
    image = np.zeros(imsize,dtype=np.float64)
    w_image = np.zeros(imsize,dtype=np.float64)

    image2 = np.zeros(imsize,dtype=np.float64)

    #normalize cloud - calculate mean quadr deviation
    deviation = 0.0
    for i in xrange(len(cloud)):
        el = cloud[i]
        deviation += (el[0]**2+el[1]**2+el[2]**2)*weights[i]**2

    deviation = deviation**0.5

    for i in xrange(len(cloud)):
        cloud[i] = cloud[i]/deviation

    #cloud - centrated

    #calculate projected coordinates of cloud

    pr_cloud = []

    xmax = 0.0
    ymax = 0.0

    for el in cloud:
        l2 = np.linalg.norm(el)**2
        y = np.dot(normal,el)
        x = (l2-y**2)**0.5

        xmax = max(xmax,x)
        ymax = max(ymax,abs(y))
        pr_cloud.append(np.array([x,y]))

    #FIXED SIZE OF IMAGE IGNORE NORMALIZATION!!!!

    xmax = 1.05*xmax
    ymax = 1.05*ymax

    stepx = xmax/imsize[0]
    stepy = 2*ymax/imsize[1]

    #image y center
    yc = imsize[1]/2.0

    for i in xrange(len(pr_cloud)):
        w = weights[i]
        el = pr_cloud[i]

        xind = int(el[0]/stepx)
        yind = int((el[1]+ymax)/stepy)
        
        #make forcefield channel
        image[xind][yind] += prop[i]*w
        w_image[xind][yind] += w

        #make shape channel 
        image2[xind][yind] += 1

    for i in xrange(imsize[0]):
        for j in xrange(imsize[1]):
            if w_image[i][j]==0:
                continue
            image[i][j] = image[i][j]/w_image[i][j]

    #unnormalized images image,image2

    return image, image2

def createMolSurfaceProperties(points_filename,mesh_index_filename,mol_filename,
        prop_type="El",prob_type="1",prob_charge=1):
    #read mesh_index, read points
    mesh_index = read_mesh_index(mesh_index_filename)
    points = read_points(points_filename)

    # wcloud - centers of triangles, weights - squares of triangles
    wcloud,weights = createWCloud(mesh_index,points)
    
    #calculate mean,centrate data with respect to weights
    wcloud,center = centrateData(wcloud,weights)
    
    #calculate pca components_ without respect to weights
    # todo: use weights
    #pca = decomposition.PCA(n_components=1)
    #pca.fit(wcloud)
    
    #pca.components_[0] - the first componenet - normal for image building
    
    #calculate mmff94 for mesh vertices
    prop = calculateMMFF94(points,mol_filename,prop_type=prop_type,
            prob_type=prob_type,prob_charge=prob_charge)
    
    #calculate mmff94 for centers of triangles (mesh faces)
    cloud_prop = calculateCloudMMFF94(prop,mesh_index)
    
    return points, mesh_index, prop, wcloud, weights, cloud_prop

def createSelectionSurfaceProperties(path,verbose=0,
        prop_type="El",prob_type="1",prob_charge=1):
    """
    """
    import sys
    filenameiter = qsar.FilenameIter(path,['points','meshidx','mol'])
    points_list = []
    mesh_index_list = []
    prop_list = []
    wcloud_list = []
    weights_list = []
    cloud_prop_list = []
    for fn in filenameiter:
        if verbose>0:
            sys.stdout.write("\r"+fn[2])
            sys.stdout.flush()
        points, mesh_index, prop, wcloud, weights, cloud_prop = \
                createMolSurfaceProperties(fn[0],fn[1],fn[2],prop_type=prop_type,prob_type=prob_type,prob_charge=prob_charge)
        points_list.append(np.asarray(points))
        mesh_index_list.append(np.asarray(mesh_index))
        prop_list.append(np.asarray(prop))
        wcloud_list.append(np.asarray(wcloud))
        weights_list.append(np.asarray(weights))
        cloud_prop_list.append(np.asarray(cloud_prop))

    return np.asarray(points_list), np.asarray(mesh_index_list), np.asarray(prop_list), \
            np.asarray(wcloud_list), np.asarray(weights_list), np.asarray(cloud_prop_list)

def createImageFromMol(points_filename,mesh_index_filename,mol_filename,imsize=(30,30)):

    #read mesh_index, read points
    mesh_index = read_mesh_index(mesh_index_filename)
    points = read_points(points_filename)

    # wcloud - centers of triangles, weights - squares of triangles
    wcloud,weights = createWCloud(mesh_index,points)
    
    #calculate mean,centrate data with respect to weights
    wcloud,center = centrateData(wcloud,weights)
    
    #calculate pca components_ without respect to weights
    # todo: use weights
    pca = decomposition.PCA(n_components=1)
    pca.fit(wcloud)
    
    #pca.components_[0] - the first componenet - normal for image building
    
    #calculate mmff94 for mesh vertices
    prop = calculateMMFF94(points,mol_filename)
    
    #calculate mmff94 for centers of triangles (mesh faces)
    cloud_prop = calculateCloudMMFF94(prop,mesh_index)
    
    #build image for this
    images = buildImage(wcloud,weights,cloud_prop,pca.components_[0],imsize)

    return images

def createImageDescriptors(path,imsize,verbose=0):
    """
    return: list of images
    """
    filenameiter = qsar.FilenameIter(path,['points','meshidx','mol'])
    images = []
    for fn in filenameiter:
        if verbose>0:
            print "filename: ",fn[2]
        im1,im2 = createImageFromMol(fn[0],fn[1],fn[2],imsize=(imsize,imsize))
        images.append((im1,im2))

    return images

def test_createImage():
    mesh_index_filename="/home/unknown/SCW/surface_test/mesh_index"
    points_filename = "/home/unknown/SCW/surface_test/points"
    mol_filename = "/home/unknown/SCW/surface_test/test.mol"
    image,image2 = createImageFromMol(points_filename,mesh_index_filename,mol_filename)
    return image1, image2

#========================================================
# Spin images (analogue of upper functions) from jupyter
#   notebook
#========================================================

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from SleepingThread import ml
import sys

def createSklearnData(points_list,prop_list):
    """
    Create data appropriate for sklearn estimators, where
        X[i] - features for single sample
    """
    # check dimensions
    if len(points_list) != len(prop_list):
        raise Exception("len(points_list) must be equal len(prop_list)")

    X = [[points_list[i], prop_list[i]] for i in xrange(len(points_list))]
    return X

def _dataFromSurface(points,props,scale=50.0):
    data = []
    for i in xrange(len(points)):
        data.append([points[i][0],points[i][1],points[i][2],props[i]/scale])
    return np.asarray(data)

def segmentSurface(points_list,prop_list,n_clusters=10,verbose=0,scale=50.0,
        random_state=None,start=-1,end=-1):
    """
    return list[ list [ <cluster label for surface point> ] ]
        list[i] - list of labels

    """
    mol_n = len(points_list)
    segment_list = []
   
    if start==-1:
        start = 0

    if end==-1:
        end=mol_n

    # create data
    for molind in xrange(start,end):
        if verbose > 0:
            sys.stdout.write("\rsegmentSurface: "+str(molind))
        points = points_list[molind]
        props = prop_list[molind]

        data = _dataFromSurface(points,props,scale=scale)
        
        #dbscan = DBSCAN()
        #dbscan.fit(data)
        #segment_list.append(dbscan.labels_)
        
        #kmeans = KMeans(n_clusters=n_clusters,random_state=random_state)
        #kmeans.fit(data)
        kmeans = KMeans(random_state=random_state)
        grs = ml.GrowSpeed(kmeans,{"n_clusters":range(max(5,len(points)/200),len(points)/30)}) 
        grs.fit(data)

        grs.drawScree()
        print "Trainer_Params: ",grs.trainer_params

        segment_list.append(np.asarray(grs.trainer.labels_))
        
    if verbose>0:
        sys.stdout.write("\n")
        
    return np.asarray(segment_list)


def _calculateProjection(points, axis):
    projection = []

    for el in points:
        l2 = np.sum(el**2)
        y = np.dot(el,axis)
        x = (l2-y**2)**0.5
        
        projection.append([x,y])
        
    return np.asarray(projection)
    

def createSpinImage1Bonds(points,axis):
    """
    points.shape = (-1,3) - list of segment vertices
        This segment already centrated and normalized
    axis.shape = (3,) - axis for projection
    """
    projection = _calculateProjection(points,axis)

    xmax = projection[:,0].max()
    ymax = max(abs(projection[:,1].min()),abs(projection[:,1].max()))

    return xmax,ymax


def createSpinImage1(points,imsize,axis,xmax=None,ymax=None):
    """
    points.shape = (-1,3) - list of segment vertices
        This segment already centrated and normalized
    axis.shape = (3,) - axis for projection
    """
    image = np.zeros(imsize,dtype=np.float64)

    projection = _calculateProjection(points,axis)

    if xmax is None:
        xmax = projection[:,0].max()
    xmin = 0.0
    xmarg = 1.05*(xmax-xmin)
   
    if ymax is None:
        ymax = max(abs(projection[:,1].min()),abs(projection[:,1].max()))
    ymarg = 1.05*ymax
    
    stepx = xmarg/imsize[0]
    stepy = 2.0*ymarg/imsize[1]
    
    for i in xrange(len(projection)):
        el = projection[i]
        xind = int(el[0]/stepx)
        yind = int((el[1]+ymarg)/stepy)
        
        image[xind][yind] += 1
    
    # normalize image
    image /= image.max()

    return image


def createSpinImage(points,imsize,add_info=False):
    image = np.zeros(imsize,dtype=np.float64)
    
    # copy points
    points = np.array(points)
    
    # centrate points
    points -= np.average(points,axis=0)
    
    # normalize points
    deviation = np.sum(points**2)**0.5
    points /= deviation
    
    # calculate main axis
    pca = PCA(n_components=1)
    pca.fit(points)
    # pca.components_[0] - main axis
    
    # calculate projection
    projection = _calculateProjection(points, pca.components_[0])
    
    xmax = 1.05*projection[:,0].max()
    ymax = 1.05*max(abs(projection[:,1].min()),abs(projection[:,1].max()))
    
    stepx = xmax/imsize[0]
    stepy = 2*ymax/imsize[1]
    
    #image y center
    yc = imsize[1]/2.0
    
    for i in xrange(len(projection)):
        el = projection[i]
        xind = int(el[0]/stepx)
        yind = int((el[1]+ymax)/stepy)
        
        image[xind][yind] += 1
    
    # normalize image
    image /= image.max()
    
    if not add_info:
        return image
    else:
        return image, points, pca.components_[0]

    return None

def _SPFromSegment(points,props,imsize):
    # points and props from one segment
    # create pair from it
    
    segm_center = np.average(points)
    
    average = np.average(props)
    sigma = np.average((props-average)**2)
    
    # create spin image from points
    image, segm_points, main_axis = createSpinImage(points,imsize,add_info=True)
    
    return [image,segm_center,average,sigma,props.min(),props.max(),segm_points,main_axis]

def _SPFromSegmentation(points,props,segmentation,imsize):
    n_clusters = segmentation.max()+1
    
    sp = []
    
    # create SP for each segment
    for segm_ind in xrange(n_clusters):
        # get segment points and props
        segm_points = points[segmentation == segm_ind]
        segm_props = props[segmentation == segm_ind]
        
        sp.append(_SPFromSegment(segm_points,segm_props,imsize))
        
    return sp

def createSP(points_list,prop_list,segment_list,verbose=0,imsize=(5,5)):
    sp_list = []
    for i in xrange(len(points_list)):        
        if verbose > 0:
            sys.stdout.write("\rmol: "+str(i))
        points = points_list[i]
        props = prop_list[i]
        segmentation = segment_list[i]
        
        sp_list.append(_SPFromSegmentation(points,props,segmentation,imsize))
       
    if verbose > 0:
        sys.stdout.write("\n")

    return sp_list


class SPType(object):
    """
    Class for feature (special points) classification
    Assign integer type for each special point

    Write class like in sklearn.cluster.KMeans
    
    Class assign types to special points in molecules
    """
    def __init__(self,n_image_clusters=10,n_prop_clusters=3):
        self.n_image_clusters = n_image_clusters
        self.n_prop_clusters = n_prop_clusters
        
        # clusterizators
        self.image_clust_ = None
        self.prop_clust_ = []
        
        self.type_dict_ = None
        
        self.labels_ = None
        
        self.n_types_ = None
        
        return
    
    def fit(self,X):
        """
        type(X) = list
        input: spdata = X[<molecule_ind>][<special_point_ind>]
        spdata[0] - image
        spdata[1] - center of cluster
        spdata[2] - average property value
        spdata[6] - segments, i.e. points and mesh_index
        spdata[7] - segm_props
        
        Algorithm:
            1) Calculate clusterization of images
            2) For each image type calculate clusters for spdata[2] scalar
            3) After steps 1,2 we have to types 
            4) Create dictionary with type pairs
            5) Assign pair to each pair in dictionary
        """
        
        # select images, props  from X
        images = []
        props = []
        for mol_sp_pts in X:
            for sp in mol_sp_pts:
                images.append(sp[0])
                props.append([sp[2]])
            
        images = np.asarray(images)
        props = np.asarray(props)
            
        # image clusterizator
        self.image_clust_ = KMeans(n_clusters=self.n_image_clusters,random_state=0,max_iter=500,n_init=20)
        self.image_clust_.fit(images.reshape(-1,images.shape[1]*images.shape[2]))
        
        # get images type
        images_type = self.image_clust_.labels_
        
        # property clusterizators
        self.prop_clust_ = [] # self.n_image_clusters*[KMeans(n_clusters=self.n_prop_clusters)]
        
        # select property for each image cluster
        for im_clust in xrange(self.n_image_clusters):
            # select properties for image type im_clust
            im_props = props[images_type == im_clust]
            
            # warning 
            if len(im_props)<self.n_prop_clusters:
                print "SPType warning: im_clust: "+str(im_clust)+" have only "+str(len(im_props))+" properties"
                
            self.prop_clust_.append( KMeans(n_clusters=min(self.n_prop_clusters,len(im_props)),random_state=0) )
            
            # fit property clusterizator
            self.prop_clust_[im_clust].fit(im_props)
            
        # add all types to dictionary
        self.type_dict_ = {}
        
        _counter = 0
        for i in xrange(self.n_image_clusters):
            for j in xrange(self.n_prop_clusters):
                self.type_dict_[(i,j)] = _counter
                _counter += 1        
                
        self.n_types_ = _counter
                
        # make labels
        self.labels_ = self.predict(X)
                
        return
    
    def predict(self,X):
        """
        Algorithm:
            1) Find image type
            2) Using classificator for this image type:
                find type of property
            3) Using dictionary:
                find type of type pair
        """
        
        sp_types = []
        for mol_sp_pts in X:
            mol_sp_types = []
            sp_types.append(mol_sp_types)
            for sp in mol_sp_pts:
                _im_type = self.image_clust_.predict([sp[0].reshape(sp[0].shape[0]*sp[0].shape[1])])[0]
                _prop_type = self.prop_clust_[_im_type].predict([[sp[2]]])[0]
                
                if (_im_type,_prop_type) in self.type_dict_:
                    mol_sp_types.append(self.type_dict_[(_im_type,_prop_type)])
                else:
                    mol_sp_types.append(-1)
        
        return np.asarray(sp_types)

    def getType(self,sp_list):
        return self.predict(sp_list)

    def getImageType(self,sp_list):
        """
        Algorithm:
            1) Find image type
            2) Using classificator for this image type:
                find type of property
            3) Using dictionary:
                find type of type pair
        """
        
        im_types = []
        for mol_sp_pts in sp_list:
            mol_im_types = []
            im_types.append(mol_im_types)
            for sp in mol_sp_pts:
                _im_type = self.image_clust_.predict([sp[0].reshape(sp[0].shape[0]*sp[0].shape[1])])[0]
                
                mol_im_types.append(_im_type)
        
        return np.asarray(im_types)


def createDescriptors(X,sptype):
    """
    create np.ndarray(len(X),sptype.n_types_)
    """
    mat = np.zeros((len(X),sptype.n_types_),dtype=np.float64)
    
    _sp_labels = sptype.predict(X)
    
    for mol_ind in xrange(len(X)):
        mol_sp_pts = X[mol_ind]
        _mol_sp_labels = _sp_labels[mol_ind]
        for sp_ind in xrange(len(mol_sp_pts)):
            _sp_label = _mol_sp_labels[sp_ind]
            if _sp_label != -1:
                mat[mol_ind][_mol_sp_labels[sp_ind]] += 1
            
    return mat


#========================================================
# Generate properties for molecules
#========================================================

import cPickle as pickle
import os

def createSurfaceProperties(sel_folder,target_filename,prop_filename,verbose=1,
        prop_type="El",prob_type="1",prob_charge=1):
    if not os.path.isfile(prop_filename):
        print "Creating data ... "
        points_list, mesh_index_list, prop_list, \
                wcloud_list, weights_list, cloud_prop_list = \
                createSelectionSurfaceProperties(sel_folder,verbose=verbose,
                        prop_type=prop_type,prob_type=prob_type,prob_charge=prob_charge)

        print " "
        print "Saving data ... "
        data_to_save = [points_list, mesh_index_list, prop_list, wcloud_list, \
                weights_list, cloud_prop_list]

        pickle.dump(data_to_save,open(prop_filename,"wb"))

    else:
        print "Loading data ... "

        # unpack data
        points_list, mesh_index_list, prop_list, \
                wcloud_list, weights_list, cloud_prop_list = \
                pickle.load(open(prop_filename,"rb"))


    # load target
    import numpy as np
    target = np.loadtxt(target_filename)

    return points_list, mesh_index_list, prop_list, \
            wcloud_list, weights_list, cloud_prop_list, target

#========================================================
# Work with parametrized special points
#========================================================

from SleepingThread.surface import Surface
import copy
import os
import cPickle as pickle

def normalize(vec):
    vec /= (np.sum(vec**2))**0.5
    return vec

def _change_CS_1(points,center,normal):
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

class MolSegmentator(object):
    def __init__(self,segment_property_type="average"):
        """
        segment_property_type = "average" | "center"
            "average": average property through whole segment
            "center": property near center of segment
        """
        self.segment_property_type = segment_property_type

        self.segment_labels = None
        
        self.segments = None
        self.segm_props = None
        self.segm_images = None
        
        self.points = None
        self.mesh_index = None
        self.props = None
        self.scales = None

        self.image = None

        # class for segmentation
        self.opt = None

        # amount of segments
        self.n_segments = None

        return

    def setSegmentPropertyType(self,segment_property_type):
        """
        """
        self.segment_property_type = segment_property_type
        return

    def setMol(self,base_filename,prop_type="El",prob_type="1",prob_charge=1.0):
        """
        This function must read:
            <base_filename>.mol
            <base_filename>.meshidx
            <base_filename>.points

        Generate force field property
        And call self.setSurface
        """

        # call setSurface
        pt_file = base_filename+".points"
        mi_file = base_filename+".meshidx"
        mol_file = base_filename+".mol"
        points,mesh_index,prop,wcloud,weights,cloud_prop = \
                createMolSurfaceProperties(pt_file,mi_file,mol_file,
                        prop_type=prop_type,prob_type=prob_type,\
                        prob_charge=prob_charge)

        self.setSurface(points,mesh_index,[prop])

        return

    def setSurface(self,points,mesh_index,props,scales=None):
        """
        points.shape = (-1,3)
        mesh_index.shape = (-1,)
        props.shape = (n_props,-1)
        scales.shape = (n_props)
        """
        # set inner variables
        self.points = np.array(points)
        self.mesh_index = np.array(mesh_index)
        self.props = np.array(props)
        self.scales = np.array(scales) 

        return

    def generateSegmentation(self,n_segments=None,scales=None,verbose=0,random_state=0):
        """
        Generate segmentation using n_segments and scales 
        """
        if scales is not None:
            self.scales = np.array(scales)


        data = _dataFromSurface(self.points,self.props[0],scale=self.scales[0])
        if n_segments is None:
            kmeans = KMeans(random_state=random_state)
            p_len = len(self.points)
            opt = ml.GrowSpeed(kmeans,{"n_clusters":range(max(5,p_len/200),p_len/30)},\
                    verbose=verbose)
            opt.fit(data)
            self.opt = opt

            self.segment_labels = opt.trainer.labels_
            self.n_segments = opt.trainer_params["n_clusters"]
        else:
            self.n_segments = n_segments
            kmeans = KMeans(n_clusters=n_segments,random_state=random_state)
            kmeans.fit(data)

            self.segment_labels = kmeans.labels_

        # function return list < list < point labels> >
        #segment_labels_list = segmentSurface([self.points],[self.props[0]],\
        #        verbose=verbose,n_clusters=n_segments,random_state=0,scale=self.scales[0])
        #self.segment_labels = segment_labels_list[0]

        # create segment surfaces
        surf = Surface(self.points,self.mesh_index)
        self.segments,self.segm_props = surf.createSegments(self.segment_labels)

        # segments: list<[ list<point>,<mesh indexes>]>
        # segm_props: list < center and normal for segment >

        return

    def drawSegmentationScores(self):
        """
        Draw scores for different amount of segments for molecule
        """
        self.opt.drawScores()
        return

    def getSegmentProperty(self,segm_ind,prtype=None):
        """
        segm_ind - segment index
        prtype = "average" | "center"
        """
        segm_surf_prop = self.props[0][self.segment_labels==segm_ind]

        if prtype is not None:
            segment_property_type = prtype
        else:
            segment_property_type = self.segment_property_type

        if segment_property_type=="center":
            return np.average(segm_surf_prop)
        elif segment_property_type=="average":
            _center = self.segm_props[segm_ind][0]
            _spoints = self.segments[segm_ind][0]
            dist = np.asarray([np.sum((el-_center)**2) for el in _spoints])
            _sort_args = np.argsort(dist)
            # get 5 nearest to center properties
            _nearest = segm_surf_prop[_sort_args[:5]]
            
            return np.average(_nearest)
        else:
            raise Exception("No such segment_property_type: "+\
                    segment_property_type)

        return 

    def getSegmentStats(self):
        """
        List statistic in table mode
        """
        from matplotlib import pyplot as plt

        fig = plt.figure()

        col_delta = []
        col_min = []
        col_max = []
        col_average = []
        col_mdisp = []
        segm_size = []
        center_prop = []
        for ind in xrange(len(self.segments)):
            # get all props for segment
            segm_surf_prop = self.props[0][self.segment_labels==ind]
            col_mdisp.append("%.2f"%metrics.mean_dispersion(segm_surf_prop)**0.5)
            _min = np.min(segm_surf_prop)
            _max = np.max(segm_surf_prop)
            col_min.append("%.2f"%_min)
            col_max.append("%.2f"%_max)
            col_delta.append("%.2d"%(_max-_min))
            col_average.append("%.2f"%np.average(segm_surf_prop))
            segm_size.append("%d"%len(segm_surf_prop))

            # get segment center property
            #_center = self.segm_props[ind][0]
            #_spoints = self.segments[ind][0]
            #dist = np.asarray([np.sum((el-_center)**2) for el in _spoints])
            #_sort_args = np.argsort(dist)
            #_nearest = segm_surf_prop[_sort_args[:5]]
            center_prop.append("%.2f"%self.getSegmentProperty(ind,"center"))

        table = np.array([col_mdisp,col_delta,\
                col_min,col_max,col_average,center_prop,segm_size]).T

        print "Columns:"
        print "1 - prop sqrt(mean dispersion)"
        print "2 - prop max-min"
        print "3 - prop min"
        print "4 - prop max"
        print "5 - prop average"
        print "6 - center prop"
        print "7 - segm size"

        plt_table = plt.table(\
                colLabels=[str(i) for i in xrange(1,table.shape[1]+1)],\
                rowLabels=[str(i) for i in xrange(len(self.segments))],\
                cellText = table,loc='top')
        #plt_table.set_fontsize(20)
        plt_table.scale(1.5,1.5)
        fig.axes[0].axis("off")
        plt.show()
        return

    def generateSegmentImages(self,imsize,immode="grayscale"):
        """
        immode - how to create image - 
            count points, or use triangle surface area
            use surface function equal to one (simple spin images)
                or use some property 
            
        immode = grayscale | colored
        """
        if immode!="grayscale":
            raise Exception("No such immode "+immode)

        self.segm_images = []
        
        x_bond = float('-inf')
        y_bond = float('-inf')
        for i in xrange(len(self.segments)):
            center,normal = self.segm_props[i]
            cur_x_bond,cur_y_bond = \
                    createSpinImage1Bonds(self.segments[i][0]-center,normal)
            x_bond = max(x_bond,cur_x_bond)
            y_bond = max(y_bond,cur_y_bond)

        for i in xrange(len(self.segments)):
            center,normal = self.segm_props[i]    
            self.segm_images.append(createSpinImage1(self.segments[i][0]-center,\
                    (imsize,imsize),normal,xmax=x_bond,ymax=y_bond))

        return

    def generateMolImage(self,imsize,immode="grayscale"):
        """
        immode - how to create image - 
            count points, or use triangle surface area
            use surface function equal to one (simple spin images)
                or use some property 
            
        immode = grayscale | colored
        """
        if immode != "grayscale":
            raise Exception("No such immode "+immode)

        self.image = createSpinImage(self.points,(imsize,imsize))

        return

    def drawSurface(self,draw_segm=False,draw_prop=False,draw_segm_centers=False,marker_size=2):
        """
        Ability:
            drawprop
            drawsegmentation

            draw segment centers
        """
        import plotly.offline as po
        import plotly.graph_objs as go
        from SleepingThread.qsar import graphics

        if draw_prop and draw_segm:
            raise Exception("Only one prop can be drawn")

        prop_list = None
        if draw_prop:
            prop_list = [self.props[0]]        
        if draw_segm:
            prop_list = [self.segment_labels]

        draw_res = graphics.drawSurfaces([self.points],[self.mesh_index],prop_list,\
                start=0,end=1,singlefigure=False,descriptions_list=["Molecule"],\
                draw=False)

        if draw_segm_centers:
            # add segment centers
            fig = draw_res[2]
            pts = []
            for el in self.segm_props:
                pts.append(el[0])
            pts = np.array(pts)
            
            sc = go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',\
                    marker=dict(size=marker_size))
            fig.append_trace(sc,1,1)

        po.iplot(draw_res[2])

        return

    def drawSegments(self,start=-1,end=-1,draw_prop=False,draw_segm_centers=False,marker_size=2,singlefigure=False,inplace=False):
        """
        """
        import plotly.offline as po
        import plotly.graph_objs as go
        from SleepingThread.qsar import graphics
        
        segm_points = []
        segm_mesh_index = []
        segm_surf_prop = []
        for ind,el in enumerate(self.segments):
            points = el[0]
            center, normal = self.segm_props[ind]
            
            if not inplace:
                #modify CS to normal
                points,center,normal = _change_CS_1(points,center,normal) 

            segm_points.append(points)
            segm_mesh_index.append(el[1])
            segm_surf_prop.append(self.props[0][self.segment_labels==ind])

        if not draw_prop:
            segm_surf_prop = None

        draw_res = graphics.drawSurfaces(segm_points,segm_mesh_index,segm_surf_prop,\
                start=start,end=end,draw=False,singlefigure=singlefigure,\
                descriptions_list=[str(i) for i in xrange(len(self.segments))])

        if draw_segm_centers:
            # add segment centers
            fig = draw_res[2]
            for i in xrange(start,end):
                center = self.segm_props[i][0]
                sc = go.Scatter3d(x=[0.0],y=[0.0],z=[0.0],\
                        mode="markers",marker=dict(size=marker_size))
                fig.append_trace(sc,i-start+1,1)
       
        po.iplot(draw_res[2])

        return draw_res[2]

    def drawSegmentImages(self,start=-1,end=-1):
        """
        """
        from SleepingThread.qsar import graphics

        graphics.drawImages(self.segm_images,start=start,end=end,\
                descriptions_list=[str(i) for i in xrange(len(self.segm_images))])

        return

    def drawMolImage(self):
        """
        """
        from SleepingThread.qsar import graphics

        graphics.drawImages([self.image],descriptions_list=["Molecule"])

        return

def getDataFromMolSegmentatorList(sgt_list,descriptors=None,verbose=0):
    """
    return data as SPGenerator1.generate
    """
    data = []
    for ind,sgt in enumerate(sgt_list):
        if verbose>0:
            sys.stdout.write("\r Process "+str(ind)+" from "+str(len(sgt_list)))

        mol_data = {}
        data.append(mol_data)

        mol_data["id"] = ind
        sp_list = []
        mol_data["sp"] = sp_list

        mol_data["labels"] = np.array(sgt.segment_labels)
        mol_data["props"] = np.array(sgt.props[0])

        if descriptors is not None:
            mol_data["descriptors"] = np.array(descriptors[ind])

        for sind,segm in enumerate(sgt.segments):
            
            sp_list.append([sgt.segm_images[sind],np.array(sgt.segm_props[sind][0]),\
                    sgt.getSegmentProperty(sind),\
                    np.array(segm),np.array(sgt.segm_props[sind])])

    if verbose>0:
        sys.stdout.write("\n")

    return data

#========================================================
#
#========================================================

class SPGenerator1(object):
    
    @staticmethod
    def _compare(arr1,arr2):

        if len(arr1)!=len(arr2):
            return False

        for ind in xrange(len(arr1)):
            if not np.all(arr1[ind]==arr2[ind]):
                return False

        return True

    def __init__(self,points_list,mesh_index_list,prop_list=None,filename=None,
            change_data=False,verbose=0):
        if len(prop_list) > 1:
            raise Exception("Multiple props not supported yet")

        self.verbose = verbose
        self.change_data = change_data
        if not change_data:
            self.points_list = np.asarray(copy.deepcopy(points_list))
            self.mesh_index_list = np.asarray(copy.deepcopy(mesh_index_list))
            self.prop_list = copy.deepcopy(prop_list)
        else:
            self.points_list = np.asarray(points_list)
            self.mesh_index_list = np.asarray(mesh_index_list)
            self.prop_list = prop_list

        self.descriptors = None

        # work with file
        self.filename = filename

        if filename is not None and os.path.isfile(filename):
            #read from file segmentation with specified scale
            points_list_loaded, \
                    mesh_index_list_loaded,\
                    self.segmentation = pickle.load(open(filename,"rb"))

            if not ( SPGenerator1._compare(points_list_loaded,points_list) and \
                    SPGenerator1._compare(mesh_index_list_loaded,mesh_index_list)):
                if verbose>0:
                    print "Loaded file has different surfaces"

                self.segmentation = {}

        else:
            self.segmentation = {}

        return

    def save(self):
        if self.filename is not None:
            pickle.dump([self.points_list,self.mesh_index_list,self.segmentation],\
                    open(self.filename,"wb"))
        else:
            raise Exception("self.filename is None")

        return

    def generate(self,n_segments,imsize=None,scale=None):
        """
        """

        if len(scale) != len(self.prop_list):
            raise Exception("unequal scale and prop_list")

        # find segmentation
        segm = self.segmentation
        if scale[0] not in segm:
            segm[scale[0]] = {}
        segm = segm[scale[0]]

        if n_segments not in segm:

            if self.verbose>0:
                print "Generate segmentation ... "

            segm[n_segments] = {}
            segm = segm[n_segments]
            
            # make segmentation
            # segment_list = list of np.ndarray
            segment_list = segmentSurface(self.points_list,self.prop_list[0],\
                    verbose=self.verbose,n_clusters=n_segments,random_state=0,scale=scale[0])

            segm["labels"] = segment_list
            segm["segments"] = []
            segm["segm_props"] = []
            segm["sp_props"] = []

            # calculate segments and its props
            # iterate over molecules surfaces
            for mol_ind in xrange(len(self.points_list)):
                if self.verbose>0:
                    sys.stdout.write("\rProcessing mol "+str(mol_ind))
                    sys.stdout.flush()

                points = self.points_list[mol_ind]
                mesh_index = self.mesh_index_list[mol_ind]

                surf = Surface(points,mesh_index)
                segments, segm_props = surf.createSegments(segment_list[mol_ind])
                # segm_props - store normals and centers for mol surface centers

                #========================================================
                # work with surface property (example: value of MMFF94)
                #========================================================
                # calculate average, min,max ... of property on surface
                sp_props = []
                surface_prop = np.array(self.prop_list[0][mol_ind])
                for segm_ind in xrange(n_segments):
                    segm_surface_prop = surface_prop[segment_list[mol_ind]==segm_ind]
                    sp_props.append([np.average(segm_surface_prop),
                        np.median(segm_surface_prop),
                        np.min(segm_surface_prop),np.max(segm_surface_prop)])

                #========================================================
                # end work with surface property
                #========================================================

                segm["segments"].append(segments)
                segm["segm_props"].append(segm_props)
                segm["sp_props"].append(sp_props)

            if self.verbose>0:
                print " "
                print "Segmentation generated"
    
        else:
            segm = segm[n_segments]

        if imsize is not None:
            # create data in format:
            # data[i] = 
            #   {"id":mol_ind,"sp":[[image,...],...],"descriptors":[<descriptors>],
            #       "labels":[<segment labels>],"props":[<MMFF94 values>]}

            # create and return Special Points list with images
            # iterate over molecules surfaces
        
            if self.verbose>1:
                print "Generate images ... "

            data = []
            
            for mol_ind in xrange(len(self.points_list)):
                segments = segm["segments"][mol_ind]
                segm_props = segm["segm_props"][mol_ind]
                # property like MMFF94
                sp_props = segm["sp_props"][mol_ind]

                mol_data = {}
                data.append(mol_data)

                mol_data["id"] = mol_ind
                sp_list = []
                mol_data["sp"] = sp_list

                mol_data["labels"] = np.array(segm["labels"][mol_ind])
                mol_data["props"] = np.array(self.prop_list[0][mol_ind])

                if self.descriptors is not None:
                    mol_data["descriptors"] = self.descriptors[mol_ind]

                # iterate over segments in molecule
                for segm_ind in xrange(n_segments):
                    # segm_ind - label for molecule surface segment
                    points,mesh_index = segments[segm_ind]
                    center,normal = segm_props[segm_ind]
                    sp_property = sp_props[segm_ind]

                    image = createSpinImage1(points-center,(imsize,imsize),normal)

                    sp_list.append([image,np.array(center),sp_property[0],
                        np.array(segments[segm_ind]),np.array(segm_props[segm_ind])])

            if self.verbose>1:
                print "End images generating"

            return data

        else:
            return

        return

    def addDescriptors(self,descr_list):
        """
        descr_list = [<descr_1>,<descr_2>]
        """
       
        if len(descr_list)==0:
            return

        descriptors = self.descriptors
        if descriptors is None:
            for mol_ind in xrange(len(self.points_list)):
                descriptors.append([])

        for descr_num in xrange(len(descr_list)):
            cur_descr = descr_list[descr_num]

            if len(cur_descr)!=len(sefl.points_list):
                raise Exception("Descriptor "+str(cur_descr)+\
                        " has inapropriate dimension")

            for mol_ind in xrange(len(self.points_list)):
                self.descriptors[mol_ind].append(cur_descr[mol_ind])

        return


def createSPGeneratorData(points_list,prop_list):
    """
    Create data only from full set
    [{"id":0,"data":[mol_points,mol_props]}]
    """
    data = []
    if len(points_list) != len(prop_list):
        raise Exception("!!!")
        
    for i in xrange(len(points_list)):
        data.append({"id":i,"data":[points_list[i],prop_list[i]]})
        
    return data

def generator_fun(mol_data,mol_id,n_sp_clusters,imsize):
    points_list = [mol_data[0]]
    prop_list = [mol_data[1]]
    # @TODO I Forget about scale in segmentSurface
    segment_list = segmentSurface(points_list,prop_list,verbose=0,n_clusters=n_sp_clusters,random_state=0)
    sp_list = createSP(points_list,prop_list,segment_list,verbose=0,imsize=(imsize,imsize))
    
    return (mol_id,sp_list[0],segment_list[0])

from sklearn.externals.joblib import Parallel, delayed

class SPGenerator(object):
    def __init__(self,generator_fun=generator_fun,filename=None,segm_filename=None,verbose=0,n_jobs=1,readonly=True):
        """
        sp[<param_ind>][<imsize>][<mol_ind>] = <sp for molecule param_ind and imsize>
        generator_fun - function, that generates sp (Special point) for single molecule
        """
        self.generator_fun = generator_fun
        self.filename = filename
        self.segm_filename = segm_filename
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.readonly = readonly
        
        if filename is not None and os.path.isfile(self.filename):
            fin = open(self.filename,"rb")
            if self.verbose > 0:
                print "SPGenerator loading data ... "
            self.sp = pickle.load(fin)
            fin.close()
        else:
            self.sp = {}
   
        if segm_filename is not None and os.path.isfile(self.segm_filename):
            fin = open(self.segm_filename,"rb")
            if self.verbose > 0:
                print "SPGenerator loading data ... "
            self.segmentation = pickle.load(fin)
            fin.close()
        else:
            self.segmentation = {}

        return
    
    def getsp(self,data,n_sp_clusters,imsize):
        """
        data[i] = {"id":0,"data":[mol_points, mol_props]}

        Output:
            result - list of [ list of special points for molecule ]
            result_segm - list of [property for points in molecule]
                property for points - segment number
        """
        result = []
        result_segm = []
      
        if not self.readonly:
            self.generate(data,n_sp_clusters,imsize)

        cur_sp = self.sp[n_sp_clusters][imsize]
        cur_segm = self.segmentation[n_sp_clusters]
            
        for mol_data in data:                
            result.append(cur_sp[mol_data["id"]])
            result_segm.append(cur_segm[mol_data["id"]])
        
        return result,result_segm

    def generate(self,data,n_sp_clusters,imsize):
        """
        """
        
        if isinstance(n_sp_clusters,list):
            for n in n_sp_clusters:
                if isinstance(imsize,list):
                    for ims in imsize:
                        self.generate(data,n,ims)
                else:
                    self.generate(data,n,imsize)
        else:   
            if self.verbose > 0:
                print "Generate (",n_sp_clusters,imsize,"). len(data): ",len(data)
            
            
            if n_sp_clusters not in self.sp:
                cur_sp1 = {}
                self.sp[n_sp_clusters] = cur_sp1
            else:
                cur_sp1 = self.sp[n_sp_clusters]

            if n_sp_clusters not in self.segmentation:
                cur_segm = {}
                self.segmentation[n_sp_clusters] = cur_segm
            else:
                cur_segm = self.segmentation[n_sp_clusters]
                
            if imsize not in cur_sp1:
                cur_sp = {}
                cur_sp1[imsize] = cur_sp
            else:
                cur_sp = cur_sp1[imsize]
                
            parallel = Parallel(n_jobs=self.n_jobs)
            works = []
            for mol_data in data:
                if mol_data["id"] not in cur_sp:
                    #cur_sp[mol_data["id"]] = self.generator_fun(mol_data["data"],n_sp_clusters,imsize)
                    works.append(delayed(self.generator_fun)(mol_data["data"],mol_data["id"],n_sp_clusters,imsize))
                    
            result = parallel(works)
            
            for mol_id,mol_sp,mol_segm in result:
                cur_sp[mol_id] = mol_sp
                cur_segm[mol_id] = mol_segm
            
            if self.verbose > 0:
                print "Generated"
            
        return
    
    def save(self,newfilename=None,new_segmfilename=None):

        if self.readonly:
            raise Exception("Readonly mode!")

        filename = None
        if newfilename is not None:
            filename = newfilename
        elif self.filename is not None:
            filename = self.filename
        else:
            raise Exception("Filename not specified")

        filename = None
        if new_segmfilename is not None:
            filename = new_segmfilename
        elif self.segm_filename is not None:
            filename = self.segm_filename
        else:
            raise Exception("Filename not specified")

        fout = open(filename,"wb")
        pickle.dump(self.segmentation,fout)
        fout.close()
        
        return
        

#========================================================
# Chemistry Dev Kit descriptors
#========================================================

def createCDKDescriptors(filenameiter=None,sel_folder=None,verbose=0,):
    """
    filenameiter [in] - object of SleepingThread.qsar.FilenameIter class
        for iteration through all molecules for descriptor calculations
    return: list of descriptors 
    """

    if filenameiter is None:
        if sel_folder is None:
            print "Error: sel_folder and filenameiter are None"

        filenameiter = qsar.FilenameIter(sel_folder,'mol')

    # CDK imports
    import jpype
    from cinfony import cdk

    descr_list = []
    for fn in filenameiter:
        if verbose>0:
            sys.stdout.write("\rCDK filename: "+str(fn))
        fis = jpype.java.io.FileInputStream(fn)
        reader = cdk.cdk.io.MDLV2000Reader(fis)
        tmpmol = reader.read(cdk.cdk.AtomContainer())
        reader.close()
        mol = cdk.Molecule(tmpmol)
        descr_list.append(mol.calcdesc())
  
    if verbose>0:
        sys.stdout.write("\n")

    # clean descr_list from nans
    from SleepingThread.data import removeNAN
    cdk_table = removeNAN(descr_list).values

    return cdk_table

#========================================================
# QSARPROJECT descriptors
#========================================================

def _readFeatures(filepath):
    data_full = []
    featuresfile = open(filepath,"r")
    for line in featuresfile:
        splitline = line.strip().split()
        row = [float(splitline[i]) for i in xrange(0,len(splitline))]
        data_full.append(row)
    data_full = np.array(data_full)

    featuresfile.close()

    return data_full


def getSDFSize(sdffilename):
    fin = open(sdffilename,"r")
    mol_amount = len(fin.read().split("$$$$"))-1
    fin.close()

    return mol_amount

def createQSARPROJECTDescriptors(inputfile,mol_amount,
        maxchainlength=4,marker="n",distmethod="none",trainset=None):
    """
    inputfile [in]: input file for selection (.qp_input file)
    marker [in]: "n" - none, "m" - multiplicity
    distmethod [in]: "none","silhouette_simple","silhouette_unique","without_distances"
    return: list of numpy arrays of features
    """
    trainsetfilepath = "/dev/shm/trainset"
    trainsetfile = open(trainsetfilepath,"w")
    #create trainset file
    if trainset is None:
        trainset = range(mol_amount)

    for mol_num in trainset:
        trainsetfile.write(str(mol_num)+" ")

    trainsetfile.close()

    featuresfilepath = "/dev/shm/features"
    #create features
    res = check_call(["./featuresmatrix_nompi_exe","--input",\
        inputfile,"--qualmode","binr2","--distmethod",distmethod,\
        "--trainsetfile",trainsetfilepath,\
        "--matrixfile",featuresfilepath,\
        "--maxchainlen",str(maxchainlength),
        "--marker",marker])
    #read features
    data_full = _readFeatures(featuresfilepath)
    return data_full

