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
import time
import torch
from sklearn.metrics.pairwise import cosine_similarity
from nilearn.plotting import plot_roi, plot_epi, show
from sklearn.cross_decomposition import PLSRegression
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as scio
import statsmodels.api as sm
import math
import glob
import argparse
import seaborn
import shutil
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import hdf5storage as hdf5
import h5py
from tqdm import tqdm
import seaborn as sb
from scipy.stats import spearmanr,pearsonr
from scipy import spatial
import random
from sklearn.metrics.pairwise import cosine_similarity
import itertools as itools
from functools import reduce
from nilearn import masking
import pickle
import json
import threading
from threading import Lock,Thread
import time,os
import multiprocessing
warnings.filterwarnings('ignore')

zs = lambda v: (v-v.mean(0))/v.std(0)
zs1 = lambda v: (v-v.mean())/v.std()

def RSA(m):
    # feature = pd.read_csv(m,header = None)
    feature = np.array(m)
    # feature = feature[:, ~np.isnan(feature).any(axis=0)]
    feature[np.isnan(feature)] = 0
    rdm = []
    a,b = feature.shape
    rdm = np.corrcoef(feature)
    rdm = np.array(rdm)
    norm_rdm = (rdm - rdm.min())/(rdm.max() - rdm.min())
    # print(norm_rdm.shape)
    # scio.savemat('/embedding_cos/'+m.split('/')[-1].split('.')[0]+'.mat', {'RDM': norm_rdm})
    return norm_rdm

def RSA_1(m):
    # feature = pd.read_csv(m,header = None)
    feature = np.array(m)
    # feature = feature[:, ~np.isnan(feature).any(axis=0)]
    feature[np.isnan(feature)] = 0
    rdm = []
    a,b = feature.shape
    rdm = cosine_similarity(feature, feature)
    rdm = np.array(rdm)
    norm_rdm = (rdm - rdm.min())/(rdm.max() - rdm.min())
    # print(norm_rdm.shape)
    # scio.savemat('/embedding_cos/'+m.split('/')[-1].split('.')[0]+'.mat', {'RDM': norm_rdm})
    return norm_rdm

def cal(norm_rdm,inp):
    x = np.dot(norm_rdm,inp)
    x = x-inp
    a,b = x.shape
    for i in range(a):
        x[i] = x[i] / (sum(abs(norm_rdm[i]))-1.0)
    corr = []
    for i in range(b):
        corr.append(pearsonr(x[:,i], inp[:,i])[0])
    return corr

def run(feature,fmri,seed,meta,roi,s):
    total_fmri = scio.loadmat(feature)['examples']
    total_feature = np.array(pd.read_csv(fmri,index_col = 0).loc[s])
    ind = list(scio.loadmat(meta)['meta'][0][0][11][0][roi][:,0])
    ind = [i-1 for i in ind]
    total_fmri = total_fmri[:,ind]
    total_fmri[np.isnan(total_fmri)] = 0
    non_zero_columns = np.any(total_fmri != 0, axis=0)
    total_fmri = total_fmri[:, non_zero_columns]
    if (total_fmri.shape[1] == 0):
        return np.zeros(1024)
    total_fmri = zs(total_fmri)
    total_feature[np.isnan(total_feature)] = 0    
    total_feature = zs(total_feature)
    np.random.seed(seed)
    np.random.shuffle(total_feature)
    np.random.seed(seed)
    np.random.shuffle(total_fmri)
    # import pdb;pdb.set_trace()
    if(total_fmri.shape[1] == 1):
        total_fmri = RSA_1(total_fmri)
    total_fmri = RSA(total_fmri)
    a = cal(total_fmri,total_feature)
    return np.array(a)

def main(args):
    # names = {'1':'DefaultA','2':'DefaultB','3':'DefaultC','4':'Language','5':'ContA','6':'ContB','7':'ContC','8':'SalVentAttnA','9':'SalVentAttnB','10':'DorsAttnA','11':'DorsAttnB','12':'Aud','13':'SomMotA','14':'SomMotB','15':'VisualA','16':'VisualB','17':'VisualC'}
    # names = {'1':'Visual1','2':'Visual2','3':'Somatomotor','4':'Cingulo-Opercular','5':'Dorsal-attention','6':'Language','7':'Frontoparietal','8':'Auditory','9':'Default','10':'Posterior-Multimodal','11':'Ventral-Multimodal','12':'Orbito-Affective'}
    feature_pars = glob.glob(args.path_feature + '/S*')
    classes_path = os.path.expanduser('/regression_lin/data/672_word.txt')
    with open(classes_path,'r',encoding = 'UTF-8') as f:
        s = f.readlines()
    s = [c.strip() for c in s]
    for feature_par in feature_pars:
        # fmri_par = '/regression_lin/data/smenatic all.csv'
        fmri_par = '/regression_lin/data/gpt2_11.csv'
        # all_list = [i for i in range(0,157)] + [i for i in range(1000,1157)]
        # import pdb;pdb.set_trace()
        # for i in all_list:
        all_roi = scio.loadmat(args.meta)['meta'][0][0][11][0].shape[0]
        for i in range(0,all_roi):
            if(os.path.exists(args.path_result+'/'+str(i+1)) is False):
                os.makedirs(args.path_result+'/'+str(i+1))
            if(os.path.exists(args.path_result+'/'+str(i+1)+'/'+feature_par.split('/')[-1]) is True):
                if(os.path.getsize(args.path_result+'/'+str(i+1)+'/'+feature_par.split('/')[-1]) != 0):
                    continue
            print(args.path_result+'/'+str(i+1)+'/'+feature_par.split('/')[-1])
            # import pdb;pdb.set_trace()
            result = run(feature_par,fmri_par,args.seed,args.meta,i,s)
            scio.savemat(args.path_result+'/'+str(i+1)+'/'+feature_par.split('/')[-1],{'ans':result})

if __name__ == '__main__':
    parser = argparse.ArgumentParser('loading path', add_help=False)
    parser.add_argument('--path_feature', type=str) 
    parser.add_argument('--path_result', type=str)
    parser.add_argument('--meta', type=str)
    parser.add_argument('--seed', type=int)
    opt = parser.parse_args()
    main(opt)