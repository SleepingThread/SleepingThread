# -*- coding: utf-8 -*-
##
#@file __init__.py
##

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

def calculateMMFF94(points,mol_file,ffname="mmff94"):
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
        prop.append(ff.calculateEl_VDW("El","1",1,point[0],point[1],point[2]))

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

def createMolSurfaceProperties(points_filename,mesh_index_filename,mol_filename):
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
    prop = calculateMMFF94(points,mol_filename)
    
    #calculate mmff94 for centers of triangles (mesh faces)
    cloud_prop = calculateCloudMMFF94(prop,mesh_index)
    
    return points, mesh_index, prop, wcloud, weights, cloud_prop

def createSelectionSurfaceProperties(path,verbose=0):
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
        points, mesh_index, prop, wcloud, weights, cloud_prop = createMolSurfaceProperties(fn[0],fn[1],fn[2])
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
import sys

def _calculateProjection(points, axis):
    projection = []

    for el in points:
        l2 = np.sum(el**2)
        y = np.dot(el,axis)
        x = (l2-y**2)**0.5
        
        projection.append([x,y])
        
    return np.asarray(projection)
    
    

def createSpinImage(points,imsize):
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
    
    return image

def _SPFromSegment(points,props,imsize):
    # points and props from one segment
    # create pair from it
    
    segm_center = np.average(points)
    
    average = np.average(props)
    sigma = np.average((props-average)**2)
    
    # create spin image from points
    image = createSpinImage(points,imsize)
    
    return [image,segm_center,average,sigma,props.min(),props.max()]

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

#========================================================
# Chemistry Dev Kit descriptors
#========================================================

def createCDKDescriptors(filenameiter,verbose=0):
    """
    filenameiter [in] - object of SleepingThread.qsar.FilenameIter class
        for iteration through all molecules for descriptor calculations
    return: list of descriptors 
    """

    # CDK imports
    import jpype
    from cinfony import cdk

    descr_list = []
    for fn in filenameiter:
        if verbose>0:
            print "CDK filename: ",fn
        fis = jpype.java.io.FileInputStream(fn)
        reader = cdk.cdk.io.MDLV2000Reader(fis)
        tmpmol = reader.read(cdk.cdk.AtomContainer())
        reader.close()
        mol = cdk.Molecule(tmpmol)
        descr_list.append(mol.calcdesc())
    
    return descr_list

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


def createQSARPROJECTDescriptors(inputfile,mol_amount,
        maxchainlength=4,marker="n",distmethod="none"):
    """
    inputfile [in]: input file for selection (.qp_input file)
    marker [in]: "n" - none, "m" - multiplicity
    return: list of numpy arrays of features
    """
    trainsetfilepath = "/dev/shm/trainset"
    trainsetfile = open(trainsetfilepath,"w")
    #create trainset file
    for mol_num in xrange(mol_amount):
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

