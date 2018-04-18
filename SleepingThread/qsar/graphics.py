# -*- coding: utf-8 -*-

"""
@todo Add description to drawSurfaces
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, rgb2hex
from matplotlib import gridspec


def drawImages(images,target=None,scale=4,width=6,cmap=plt.get_cmap("gray_r"),textcolor='black',fontsize=12,l=-1,filename=None,descriptions_list=None,start=-1,end=-1):
    """
    images - list of np.array((image_height,image_width))
    target - labels for each image
    """
    if l == -1:
        l = len(images)

    if start == -1:
        start = 0
    if end == -1:
        end = l

    if target is None:
        target_sort = False
        target = np.array(["None" for i in xrange(l)])
    else:
        target_sort = True

    if descriptions_list is None:
        descriptions_list = np.array(["" for i in xrange(l)])

    images = np.asarray(images)
    descriptions_list = np.asarray(descriptions_list)
    target = np.asarray(target)

    if target_sort:
        ids = np.argsort(target)
        target = target[ids]
        descriptions_list = descriptions_list[ids]
        images = images[ids]

    # amount of images for drawing
    dn = end-start

    fig = plt.figure(figsize=(width,width*dn))
    imagemax = np.max(images)
    imagemin = np.min(images)
    
    norm = Normalize(vmin=imagemin,vmax=imagemax)
   
    grid = gridspec.GridSpec(scale*dn+1,1)
    axarr = []
    axarr.append(fig.add_subplot(grid[0:scale,0]))
    axarr.extend([fig.add_subplot(grid[scale*i:scale*(i+1),0],sharex=axarr[0]) for i in xrange(1,dn)])
    axarr.append(fig.add_subplot(grid[scale*dn:scale*dn+1]))
    for i in xrange(start,start+dn):
        axarr[i].imshow(images[i],norm=norm,cmap=cmap)
        text = str(target[i])+" | "+str(i)+" | "+str(descriptions_list[i])
        axarr[i].text(0.5,0.95,text,transform=axarr[i].transAxes,color='black',fontsize=12,weight='bold')
    
    mpl.colorbar.ColorbarBase(axarr[-1],cmap=cmap,norm=norm,orientation='horizontal')
    fig.tight_layout()
   
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename,dpi=96)
    
    return

def drawSurfaces(points_list,mesh_index_list=None,prop_list=None,target=None,cmap=plt.get_cmap('jet'),start=-1,end=-1,jupyter_notebook=True,filename="test",verbose=0,fileformat="detailed",
        scattermode=False,singlefigure=False,draw=True,descriptions_list=None):
    """
    fileformat = "detailed" | "raw"
    """
   
    import sys
    from plotly import tools
    import plotly.offline as po
    import plotly.graph_objs as go
    from plotly.offline import init_notebook_mode

    n = len(points_list)

    if target is None:
        target_sort = False
        target = np.array(["None" for i in xrange(n)])
    else:
        target_sort = True
    
    if prop_list is None:
        prop_list = np.array([1 for i in xrange(n)])
    else:
        # check dims
        for i in xrange(n):
            if len(prop_list[i])!=len(points_list[i]):
                raise Exception("prop_list and points_list different dims")

    if mesh_index_list is None:
        if not scattermode:
            raise Exception("mesh_index_list = None with scattermode must be True")
        mesh_index_list = np.array([[1] for i in xrange(n)])

    if descriptions_list is None:
        descriptions_list = np.array(["" for i in xrange(n)])

    if jupyter_notebook:
        init_notebook_mode(connected=True)
     
    if start == -1:
        start = 0
    if end == -1:
        end = len(target)
        
    #draw number
    dn = end-start
    
    # sort data if target specified
    if target_sort:
        ids = np.argsort(target)
        target = target[ids]
        points_list = points_list[ids]
        mesh_index_list = mesh_index_list[ids]
        prop_list = prop_list[ids]
        descriptions_list = descriptions_list[ids]
    
    # create color map for prop_list
    # find min,max
    propmin = float("+inf")
    propmax = float("-inf")
    minind = -1
    maxind = -1
    for i in xrange(n):
        props = prop_list[i]
        if propmin>np.min(props):
            minind=i
            propmin=np.min(props)
        if propmax<np.max(props):
            maxind=i
            propmax = np.max(props)
   
    if verbose > 0:
        print "min/max: ",propmin,propmax
        print "inds: min/max",minind,maxind
    
    propmin = float(propmin)
    propmax = float(propmax)
   
    if propmin!=propmax:
        vertex_colors = [[rgb2hex(cmap((val-propmin)/(propmax-propmin))) for val in prop_list[i]] for i in xrange(start,end)]
    else:
        vertex_colors = [[rgb2hex(cmap(0.5)) for val in points_list[i]] for i in xrange(start,end)]

    if not singlefigure:
        titles = [str(target[i])+" | "+str(i)+" | "+descriptions_list[i] \
                for i in xrange(start,end)]
        fig = tools.make_subplots(dn,1,\
                                  subplot_titles=titles,\
                                  specs=dn*[[{'is_3d':True}]],shared_xaxes=True,\
                                  vertical_spacing=0.1/n,print_grid=False)
    else:
        fig = tools.make_subplots(1,1,\
                                  subplot_titles="",\
                                  specs=[[{'is_3d':True}]],shared_xaxes=True,\
                                  vertical_spacing=0.1/n,print_grid=False)
 
    for i in xrange(start,end):
        pts = points_list[i]
        sys.stdout.write("\r"+str(i)+" mol")
        if not scattermode:
            ids = mesh_index_list[i].reshape((-1,3))
            text = str(target[i])+" | "+str(i)+" | "+descriptions_list[i]
            mesh = go.Mesh3d(x = pts[:,0],y=pts[:,1],z=pts[:,2],\
                                       i=ids[:,0],j=ids[:,1],k=ids[:,2],\
                                       vertexcolor=vertex_colors[i-start],\
                                       text=text, name="")
            if not singlefigure:
                fig.append_trace(mesh,i-start+1,1)
            else:
                fig.append_trace(mesh,1,1)
        else:
            sc = go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode="markers",marker=dict(size=2))
            if not singlefigure:
                fig.append_trace(sc,i-start+1,1)
            else:
                fig.append_trace(sc,1,1)
       
    if not singlefigure:
        fig['layout'].update(height=dn*400, width=600, title='Mols')
    else:
        fig['layout'].update(height=400,width=600, title="Mols")
    
    if jupyter_notebook:
        #po.iplot(fig,image="png",filename="test")
        if draw:
            po.iplot(fig)
    else:
        if draw:
            if fileformat == "detailed":
                po.plot(fig,filename+"("+str(start)+"-"+str(end)+").html")
            elif fileformat == "raw":
                po.plot(fig,filename)

        #po.image.save_as(fig,filename="test.png")
                         
    return propmin, propmax, fig

def drawColorMap(vmin,vmax,figsize=(12,1),cmap=plt.get_cmap('jet'),filename=None):
    fig = plt.figure(figsize=(12,1))
    norm = Normalize(vmin=vmin,vmax=vmax)
    mpl.colorbar.ColorbarBase(fig.gca(),cmap=cmape,norm=norm,orientation='horizontal')
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename,dpi=96)

    return
