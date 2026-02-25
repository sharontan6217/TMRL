import os
import warnings; warnings.simplefilter('ignore')
import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
import math
from pathlib import Path
import torch
import zipfile
import librosa
from portable_m2d import PortableM2D
import pandas as pd
import numpy as np
import random
import gc
import yaml
import csv
import glob
import utils
from utils import utils
import model
from model import tagging, infer
import evaluation


from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, hamming_loss,jaccard_score,roc_auc_score,roc_curve, precision_recall_curve,auc, f1_score,silhouette_score,normalized_mutual_info_score, adjusted_rand_score
from IPython.display import display, Audio
#import plotnine
#from plotnine import *
import matplotlib.pyplot as plt
import datetime
from datetime import datetime,time
import argparse



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='./sound_datasets/rare_sound_event/eval/audio/', help = 'directory of the original data.' ) 
    parser.add_argument('--graph_dir',type=str,default='./graph/', help = 'directory of graphs.' )
    parser.add_argument('--output_dir',type=str,default='./output/', help = 'directory of outputs for AEC and ASC task.')
    parser.add_argument('--log_dir',type=str,default='./log/', help = 'directory of the transaction logs.')
    parser.add_argument('--noise_factor',type=float,default=0, help = 'noise to be added to original files for experiments')
    parser.add_argument('--classification_type',type=str,default='event', help = 'three options: event, environment, or both')
    opt = parser.parse_args()
    return opt


if __name__=="__main__":
    start = datetime.now()
    project_dir=os.getcwd()
    os.chdir(project_dir)
    opt = get_parser()
    data_dir = opt.data_dir
    output_dir = opt.output_dir
    graph_dir = opt.graph_dir
    log_dir = opt.log_dir
    noise_factor = opt.noise_factor
    event_dir = output_dir+'event/'
    env_dir =output_dir+'env/'
    if os.path.exists(graph_dir)==False:
        os.makedirs(graph_dir)
    if os.path.exists(event_dir)==False:
        os.makedirs(event_dir)
    if os.path.exists(env_dir)==False:
        os.makedirs(env_dir)
    if os.path.exists(log_dir)==False:
        os.makedirs(log_dir)
 

    meta_path = "./sound_datasets/rare_sound_event/meta"

    model_event_name = 'all-mpnet-base-v2'
    model_environment_name = 'all-mpnet-base-v2'
    #model_event = SentenceTransformer('stsb-roberta-large')
    model_event = SentenceTransformer(model_event_name)
    #model_similarity = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #model_similarity = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    #model_similarity = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    model_environment = SentenceTransformer(model_environment_name)
    gc.collect()
    classes = pd.read_csv('class_labels_indices.csv').sort_values('mid').reset_index()
    #classes[:3]
    '''
    with zipfile.ZipFile("m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    '''

    
    #model = PortableM2D(weight_file='m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d/weights_ep69it3124-0.47929.pth', num_classes=527)
    model = PortableM2D(weight_file='m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly/weights_ep67it3124-0.48558.pth',num_classes=527)
    timesequence=str(start)[-6:]
    print(timesequence)
    print(str(datetime.now()))
    df_meta= utils.metaLoad(meta_path)


    print(df_meta)
    print(df_meta.columns)

    #files = np.random.choice(files, size=len(files), replace=False)
    event_list=['babycry','gunshot', 'glassbreak']
    envnt_list=['beach','bus','cafe/restaurant','car','city_center','forest_path','grocery_store','home','library','metro_station','office','park','residential_area','train','tram']
    #random_envnt = random.randrange(0,len(envnt_list_total)-2) 
    #envnt_list=['beach','bus','cafe/restaurant','forest_path']  
    #envnt_list=['train','home','cafe/restaurant','residential_area']   
    classification_type = opt.classification_type
    df_envnt=pd.DataFrame()
    for envnt in envnt_list:
        df_meta_event = df_meta[df_meta['bg_classname'] == envnt]
        df_envnt = pd.concat([df_envnt,df_meta_event],axis=0,ignore_index=True)

    df_envnt = df_envnt.sort_values(by='wav_name',ascending=True)
    df_envnt = df_envnt.reset_index()
    matrix_total=pd.DataFrame()
    actual_event_total=np.array([])
    predict_event_total=np.array([])
    wav_name_total = np.array([])
    files_per_batch = 2
    for i in range(1):
        #random_int = random.randrange(0,489)
        random_int = 0
        files_babycry = list(Path(data_dir+'babycry').glob('*.wav'))[random_int:random_int+files_per_batch]
        files_gunshot = list(Path(data_dir+'gunshot').glob('*.wav'))[random_int:random_int+files_per_batch]
        files_glassbreak = list(Path(data_dir+'glassbreak').glob('*.wav'))[random_int:random_int+files_per_batch]
        files = [*files_babycry,*files_glassbreak,*files_gunshot]
        #files = [*files_babycry]
        print(files)
        wav_list =[]
        topk_=[]
        df_representation = pd.DataFrame()
        for j in range(len(df_envnt)):   
            for fn in files:
                fn_ = str(fn).split('\\')[-1]
                if fn_ == df_envnt['wav_name'][j]:
                    event = df_envnt['event_actual'][j]
                    min_duration,duration,target_sr = utils.computeDuration(classification_type,df_meta,event)
                    top_classes, top_values, secs, sub_representation=tagging.show_topk_sliding_window(classes,duration,min_duration,target_sr,model, fn,opt)
                    df_representation = pd.concat([df_representation,sub_representation],axis=0,ignore_index=True )
                    wav_list.append(fn_)
                    j+=1
        df_representation = df_representation.merge(df_envnt,how='left',on=['wav_name'],suffixes=('','_'+str('subclass')))
        df_representation = df_representation.reset_index()
        #print('merged original data is: ',df_representation)
        df_representation.to_csv(output_dir+model_event_name+'_representation_orig_'+str(i)+'.csv')
        #df_representation = infer.env_sounds_simple(envnt_list,wav_list,df_representation,model_environment)
        #df_representation = infer.env_sounds(envnt_list,wav_list,df_representation,model_environment)
        if classification_type=='event':
            df_representation = infer.event_sounds(event_list,df_representation,model_event)
            df_representation.to_csv(event_dir+model_event_name+'_representation_result_'+str(i)+'_noisefactor_'+str(noise_factor)+'.csv')
            try:
                matrix, actual_event,predict_event,wav_name_array= evaluation.evaluationMatrix(df_representation,classification_type,event,random_int,opt)
                matrix_total=pd.concat([matrix_total,matrix],axis=0)
                actual_event_total = np.concatenate([actual_event_total,actual_event],axis=0)
                predict_event_total = np.concatenate([predict_event_total,predict_event],axis=0)
                wav_name_total = np.concatenate([wav_name_total,wav_name_array],axis=0)
            except Exception as e:
                print(e)
                pass
        elif classification_type=='environment':
            df_representation = infer.env_sounds(envnt_list,wav_list,df_representation,model_environment)
            df_representation.to_csv(env_dir+model_environment_name+'_env_representation_result_'+str(i)+'_noisefactor_'+str(noise_factor)+'.csv')
        elif classification_type=='both':
            df_representation = infer.event_sounds(event_list,df_representation,model_event)
            df_representation.to_csv(event_dir+model_event_name+'_event_representation_result_'+str(i)+'_noisefactor_'+str(noise_factor)+'.csv')
            try:
                matrix, actual_event,predict_event,wav_name_array= evaluation.evaluationMatrix(df_representation,classification_type,event,random_int,opt)
                matrix_total=pd.concat([matrix_total,matrix],axis=0)
                actual_event_total = np.concatenate([actual_event_total,actual_event],axis=0)
                predict_event_total = np.concatenate([predict_event_total,predict_event],axis=0)
                wav_name_total = np.concatenate([wav_name_total,wav_name_array],axis=0)
            except Exception as e:
                print(e)
                pass
            df_representation = infer.env_sounds(envnt_list,wav_list,df_representation,model_environment)
            df_representation.to_csv(env_dir+model_environment_name+'_env_representation_result_'+str(i)+'_noisefactor_'+str(noise_factor)+'.csv')
        i+=1

    
    
    matrix_total.to_csv(output_dir+model_event_name+'_matrix_total_'+event+'_'+timesequence+'.csv')

    output = pd.DataFrame()
    output['classification_type']=classification_type
    output['wav_name']=wav_name_total
    output['actual_event']=actual_event_total
    output['predict_event']=predict_event_total
    output.to_csv(output_dir+'output_'+event+'_'+timesequence+'.csv')
    end = datetime.now()
    print(start,end)
    #fig = evaluation.visualize(matrix_total,actual_event_total,predict_event_total,event,'event',opt)
    '''
    for fn in files:
        show_topk_sliding_window(classes, model, fn)

        show_topk_for_all_frames(classes, model, files[0])
    '''