import os
import csv
import warnings
from nilearn import datasets, plotting
from nilearn.image import index_img, mean_img
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn
import re
from nilearn.plotting import plot_roi, plot_epi, show
from sklearn.cross_decomposition import PLSRegression
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats
from  math  import  sqrt
import glob
import shutil
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import scipy.io as scio
import hdf5storage as hdf5
import h5py
from itertools import combinations
import random
import math
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import random
from scipy import stats
import itertools as itools
from functools import reduce
from nilearn import masking
from scipy.stats import spearmanr,pearsonr
zs = lambda v: (v-v.mean(0))/v.std(0)
warnings.filterwarnings('ignore')

classes_path = os.path.expanduser('feature.txt')
with open(classes_path,'r',encoding = 'utf-8') as f:
    s = f.readlines()
word = [c.strip() for c in s]

models = glob.glob('/network_to_semantic_revised/*')
models = sorted(models, key=lambda x: x.split('/')[-1])

data = []
for model in models:
    mid = []
    files = glob.glob(model+'/*')
    for file in files:
        test = scio.loadmat(file)['ans'][0]
        mid.append(test)
    mid = np.average(mid,axis = 0)
    data.append(mid)
    
data = np.array(data)
models = [i.split('/')[-1] for i in models]

data_al = data.mean(axis = 1)
mid = []
for i in range(len(models)):
    if(data_al[i] < -100):
        mid.append(i)
data = np.delete(data, mid, axis=0)
models = [element for idx, element in enumerate(models) if idx not in mid]
# data[data < 0] = 0

fig, ax = plt.subplots(figsize=(20,4))
# color map
# cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
# plot heatmap
fig = sns.heatmap(data[:,0:30], annot=True, fmt=".2f",  cmap='Blues',vmin=0, vmax=0.5,annot_kws={"fontsize":11},cbar_kws={"shrink": .6},linewidths=2)
yticks = models
xticks = word[0:30]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0,fontsize = 14)
plt.xticks(plt.xticks()[0], labels=xticks,rotation=90,fontsize = 14)
heatmap = fig.get_figure()
heatmap.savefig('/picture_revised/kong_feature1', dpi = 400,bbox_inches='tight')

# 计算脑的相关性，并将图画出来 前30个feature
zs1 = lambda v: (v-v.mean())/v.std()
classes_path = os.path.expanduser('feature.txt')
with open(classes_path,'r',encoding = 'utf-8') as f:
    s = f.readlines()
word = [c.strip() for c in s]

models = glob.glob('/network_to_semantic_revised/*')
models = sorted(models, key=lambda x: x.split('/')[-1])

data = []
for model in models:
    mid = []
    files = glob.glob(model+'/*')
    for file in files:
        test = scio.loadmat(file)['ans'][0]
        mid.append(test)
    mid = np.average(mid,axis = 0)
    data.append(mid)
    
data = np.array(data)

models = [i.split('/')[-1] for i in models]

data_al = data.mean(axis = 1)
mid = []
for i in range(len(models)):
    if(data_al[i] < -100):
        mid.append(i)
data = np.delete(data, mid, axis=0)
models = [element for idx, element in enumerate(models) if idx not in mid]
# data[data < 0] = 0

fig, ax = plt.subplots(figsize=(20,4))
# color map
# cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
# plot heatmap
fig = sns.heatmap(data[:,30:], annot=True, fmt=".2f", cmap='Blues', vmin=0, vmax=0.5,annot_kws={"fontsize":11},cbar_kws={"shrink": .6},linewidths=2)
yticks = models
xticks = word[30:]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0,fontsize = 14)
plt.xticks(plt.xticks()[0], labels=xticks,rotation=90,fontsize = 14)
heatmap = fig.get_figure()
heatmap.savefig('/picture_revised/kong_feature2', dpi = 400,bbox_inches='tight')

