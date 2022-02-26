import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import sys

#Functions
def InitData(Nbirds = 20, box = (0,50), dim = 3):
    """
    Nbirds : int
        Number of birds
    box : tuple
        tuple containing min and max value
        of where the birds fly
    """
    #Position of the birds
    position = np.random.uniform(low = box[0], high = box[1], size = (Nbirds,dim))
    #Heading of the birds
    direction = np.random.normal(loc=0, scale=10, size=(Nbirds,dim))
    #Concatenate data
    data = np.concatenate([position,direction], axis = 1)
    #Normalise directional data
    data = DataNorm(data = data, dim = dim)
    #Return data
    return data
def VecNorm(vec, dim = 3):
    """
    Calculate norm of the direction
    of a vector
    vec : numpy.ndarray
        6 entries, first 3 is position
        last three is
    """
    tmp_vec = np.copy(vec)
    length = np.linalg.norm(vec[dim:])
    tmp_vec[dim:] = vec[dim:] / length
    return tmp_vec

def VecDist(vec1, vec2, dim = 3):
    #Relevant parts of the vectors
    v1 = vec1[:dim]
    v2 = vec2[:dim]
    return np.linalg.norm(v1-v2)

def cpos(data, dim):
    """
    Correct the position of birds
    """
    #Correct the birds
    #stay inside the [0,50]^3 box
    tmp_data = np.copy(data)
    tmp_data[:,:dim] = tmp_data[:,:dim]%50
    return tmp_data

def DataNorm(data, dim = 3):
    """
    Normalise headings of the birds
    """
    #normalise the directional part of data
    tmp_data = np.copy(data)
    tmp_data[:,dim:] = tmp_data[:,dim:]/np.linalg.norm(tmp_data[:,dim:], axis = 1).reshape(-1,1)
    return tmp_data

def Cvecs(idx, data, dist = 10, dim = 3):
    #Return closest vectors of 
    #the idx'th data point
    
    #Copy of data
    tmp_data = np.copy(data)
    #vector of intrest
    vec = data[idx]
    #relevant data
    rel_data = (data[:,:dim]-vec[:dim])
    #distances to rest of data
    lengths = np.linalg.norm(rel_data, axis = 1)
    #indices of the data within a distances of dist
    indices = np.where(lengths < dist)[0]
    indices = indices[indices != idx]
    #return vectors that are "dist" away from 
    #our vector of interest
    return tmp_data[indices]

def Cdir(idx, data, dist = 10, avoidDist = 2, dim = 3):
    #Change directions
    
    #direction of closest birds
    vec = np.copy(data[idx])
    vecs = Cvecs(idx, data, dist = dist)
    
    #Initial directions
    avgDir = np.zeros(dim)
    cdir = np.zeros(dim)
    
    #If there are birds close 
    #Update the directions
    if len(vecs)>0:
        #Average direction
        avgDir = np.sum(vecs[:,dim:], axis = 0)
    
        #center point
        cpoint = np.mean(vecs[:,:dim], axis = 0)
        cdir = cpoint-vec[:dim]
        
        #Normalise directional vector
        cdir = cdir/np.linalg.norm(cdir)
        avgDir = avgDir/np.linalg.norm(avgDir)
    
    
    #   Steer away from birds too close
    #Birds that are too close (closer than avoidDist)
    too_close = Cvecs(idx, data, dist = avoidDist, dim = dim)[:,:dim]
    #Initial direction to steer away
    avoidDir = np.zeros(dim)
    #Change avoidance direction if any birds are too close
    if len(too_close)>0:
        avoidDir = -np.nanmean(too_close-vec[:dim], axis = 0)
        #Normalise direction
        avoidDir = avoidDir/np.linalg.norm(avoidDir)
    
    total_center = np.mean(data[:,dim:], axis = 0)
    total_center = total_center/np.linalg.norm(total_center)
    random_dir = np.random.normal(0,5, size = dim)
    random_dir = random_dir/np.linalg.norm(random_dir)
    #New direction
    if len(vecs)>0:
        nDir = np.average([cdir, avgDir, avoidDir, random_dir], axis=0, weights=[1/10,3/10,4/10, 2/10])
        #nDir = np.mean([1.5*cdir, 1.5*avgDir, avoidDir, 0.3*random_dir], axis = 0)
        nDir = nDir/np.linalg.norm(nDir)
        return nDir
    else:
        nDir = vec[dim:]
        return nDir

def update_all_directions(data, dist = 15, avoidDist = 2, dim = 3):
    """
    Update the direction of all birds
    """
    tmp_data = np.copy(data)
    #For each bird update the direction/heading
    for i in range(len(data)):
        tmp_data[i,dim:] = Cdir(idx = i, data = data, dist = dist, avoidDist = avoidDist, dim = dim)
        
    #Normalise direction
    tmp_data = DataNorm(tmp_data, dim = dim)
    return tmp_data

def takeStep(data, stepsize = 1, dim = 3):
    """
    Take a step in the direction of the directional vector
    """
    tmp_data = np.copy(data)
    tmp_data[:,:dim] = tmp_data[:,:dim]+stepsize*tmp_data[:,dim:]
    #correction the position
    tmp_data = cpos(tmp_data, dim)
    return tmp_data

def runSim():
    return None
def Birdplot(df, dim):
    if dim ==3:
        fig = px.scatter_3d(df, x="x", y="y",z = "z" ,animation_frame="t",
          range_x=[0,50], range_y=[0,50], range_z = [0,50], color = "z")
        fig.update_traces(marker_size = 3)
    if dim ==2:
        fig = px.scatter(df, x="x", y="y",animation_frame="t",
          range_x=[0,50], range_y=[0,50])
        fig.update_traces(marker_size = 5)
    fig.show()
    return fig



#Example
Nbirds = 200
dim = 2
data = InitData(Nbirds = Nbirds, box = (0,50), dim = dim)
#Number of iterations to take
Niter = 1000
#Copy of the inital data
new_data = np.copy(data)

#Pandas dataframe
base_dataframe = pd.DataFrame()
for j in range(Niter):
    sys.stdout.write('\r'+"Iteration: "+str(j+1)+"/"+str(Niter))
    #Update directions
    new_data = update_all_directions(new_data, dist = 3, avoidDist = 1, dim = dim)
    #Take a step in new direction
    new_data = takeStep(new_data, stepsize = 0.1, dim = dim)
    #Correction position
    new_data = cpos(new_data, dim)
    #Save as dataframe
    if dim == 3:
        df = pd.DataFrame(new_data, columns = ["x","y","z","u","v", "w"])
    if dim == 2:
        df = pd.DataFrame(new_data, columns = ["x","y","u","v"])
    df["t"] = j
    #Concatenate dataframes
    base_dataframe = pd.concat([base_dataframe, df])

Birdplot(base_dataframe, dim)