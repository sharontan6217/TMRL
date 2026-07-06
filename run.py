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
import yaml

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, hamming_loss,jaccard_score,roc_auc_score,roc_curve, precision_recall_curve,auc, f1_score,silhouette_score,normalized_mutual_info_score, adjusted_rand_score
from IPython.display import display, Audio

#import plotnine
#from plotnine import *
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import time
import argparse



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='./sound_datasets/rare_sound_event/eval/audio/', help = 'directory of the original data.' ) 
    #parser.add_argument('--data_dir',type=str,default='./sound_datasets/data/eval21/ground_truth', help = 'directory of the original data.' ) 
    #parser.add_argument('--data_dir',type=str,default='./sound_datasets/TAU Urban Acoustic Scenes/TAU-urban-acoustic-scenes-2020-mobile-development/audio/', help = 'directory of the original data.' ) 
    parser.add_argument('--pretrained_m2d',type=str,default='m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly/weights_ep67it3124-0.48558.pth', help = 'directory of the pretrained m2d model.' ) 
    #parser.add_argument('--pretrained_m2d',type=str,default='m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d/weights_ep69it3124-0.47929.pth', help = 'directory of the pretrained m2d model.' ) 
    parser.add_argument('--graph_dir',type=str,default='./graph/event/', help = 'directory of graphs.' )
    parser.add_argument('--output_dir',type=str,default='./output/eventd/', help = 'directory of outputs for AEC and ASC task.')
    parser.add_argument('--log_dir',type=str,default='./log/event/', help = 'directory of the transaction logs.')
    parser.add_argument('--noise_factor',type=float,default=0, help = 'noise to be added to original files for experiments')
    parser.add_argument('--classification_type',type=str,default='event', help = 'three options: event, environment, or both')
    opt = parser.parse_args()
    return opt

class experiment():
    def config():
        with open ('config/tmrl.yaml','r') as f:
            config_tmrl = yaml.safe_load(f)
        meta_path = config_tmrl['data']['metafiles']
        model_event_name = config_tmrl['model_infer']['model_event_name']
        model_environment_name = config_tmrl['model_infer']['model_event_name']
        files_per_batch = config_tmrl['train']['files_per_batch']
        return  meta_path,model_event_name, model_environment_name,  files_per_batch
    
    def run(files_per_batch,meta_path,model_event,model_environment,model_m2d,envnt_list,event_list):


        classification_type = opt.classification_type
        #df_envnt = df_envnt.sort_values(by='wav_name',ascending=True)
        #df_envnt = df_envnt.reset_index()
        df_representation_environ = pd.DataFrame() 
        df_representation_event = pd.DataFrame()
        df_representation_tagging=pd.DataFrame()
        #matrix_total=pd.DataFrame()
        #actual_event_total=np.array([])
        #predict_event_total=np.array([])
        #wav_name_total = np.array([])
        wav_list =[]
        for i in range(files_per_batch):
            random_int = random.randrange(0,150)
            #random_int = 1
            files_babycry = list(Path(data_dir+'babycry').glob('*.wav'))[random_int:random_int+1]
            files_gunshot = list(Path(data_dir+'gunshot').glob('*.wav'))[random_int:random_int+1]
            files_glassbreak = list(Path(data_dir+'glassbreak').glob('*.wav'))[random_int:random_int+1]
            files = [*files_babycry,*files_glassbreak,*files_gunshot]
            #files = [*files_glassbreak]

            print(files)
              
            for fn in files:
                fn_ = str(fn).split('\\')[-1]
                print(fn,fn_)
                for target_event in event_list:
                    event = target_event
                    sr,  target_sr ,lower_db ,higher_db ,k ,hop, class_dir = tagging.config()
                    duration, min_duration = utils.computeDuration(classification_type,pretrain_meta,event)
                    top_classes, top_values, secs, sub_representation=tagging.show_topk_sliding_window(class_dir, sr, duration,min_duration,target_sr,lower_db,higher_db,model_m2d, fn, opt, k, hop)
                    #top_classes, top_values, secs, sub_representation=tagging.show_topk_for_all_frames(class_dir, sr, duration,min_duration,target_sr,lower_db,higher_db,model_m2d, fn, opt, k)
                    df_representation_tagging = pd.concat([df_representation_tagging,sub_representation],axis=0,ignore_index=True )
            i+=1

        print(df_representation_tagging)
        print(df_representation_tagging.columns)


        #print('merged original data is: ',df_representation)
        df_representation_tagging.to_csv(output_dir+model_event_name+'_representation_orig_'+timesequence+'_'+str(random_int)+'_'+str(i)+'.csv')
        #df_representation = infer.env_sounds_simple(envnt_list,wav_list,df_representation,model_environment)
        #df_representation = infer.env_sounds(envnt_list,wav_list,df_representation,model_environment)

        meta= utils.metaLoad(meta_path)

        print(meta)
        print(meta.columns)


        


        if classification_type=='event':
            df_representation_event = infer.event_sounds(event_list,df_representation_tagging,model_event)
            df_representation_event.to_csv(event_dir+model_event_name+'_event_representation_result_'+timesequence+'_'+str(random_int)+'_'+str(i)+'_noisefactor_'+str(noise_factor)+'.csv')                
            df_meta=pd.DataFrame()
            for envnt in envnt_list:
                df_meta_event = meta[meta['bg_classname'] == envnt]
                df_meta = pd.concat([df_meta,df_meta_event],axis=0,ignore_index=True)
            df_representation_event = df_representation_event.merge(df_meta,how='left',on=['wav_name'],suffixes=('','_'+str('subclass')))
            df_representation_event = df_representation_event.reset_index()
            matrix_total, actual_event_total,predict_event_total,wav_name_total= evaluation.evaluationMatrix(df_representation_event,classification_type,event_list,random_int,opt)

        elif classification_type=='environment':
            number_of_predicted_environments = infer.config()
            df_representation_environ = infer.env_sounds(envnt_list,wav_list,df_representation_tagging,model_environment,number_of_predicted_environments)
            df_representation_environ.to_csv(env_dir+model_environment_name+'_env_representation_result_'+timesequence+'_'+str(random_int)+'_'+str(i)+'_noisefactor_'+str(noise_factor)+'.csv')
            df_meta=pd.DataFrame()
            for envnt in envnt_list:
                df_meta_event = meta[meta['bg_classname'] == envnt]
                df_meta = pd.concat([df_meta,df_meta_event],axis=0,ignore_index=True)
            df_representation_environ = df_representation_environ.merge(df_meta,how='left',on=['wav_name'],suffixes=('','_'+str('subclass')))
            df_representation_environ = df_representation_environ.reset_index()
            matrix_total, actual_event_total,predict_event_total,wav_name_total= evaluation.evaluationMatrix(df_representation_environ,classification_type,event_list,random_int,opt)

        elif classification_type=='both':
            number_of_predicted_environments = infer.config()
            df_representation_event = infer.event_sounds(event_list,df_representation_tagging,model_event)
            df_representation_event.to_csv(event_dir+model_event_name+'_event_representation_result_'+timesequence+'_'+str(random_int)+'_'+str(i)+'_noisefactor_'+str(noise_factor)+'.csv')
            df_representation_environ = infer.env_sounds(envnt_list,wav_list,df_representation_tagging,model_environment,number_of_predicted_environments)
            df_representation_environ.to_csv(env_dir+model_environment_name+'_env_representation_result_'+timesequence+'_'+str(random_int)+'_'+str(i)+'_noisefactor_'+str(noise_factor)+'.csv')
            df_representation = df_representation_event.merge(df_representation_environ,how='outer',on='wav_name')
            df_meta=pd.DataFrame()
            for envnt in envnt_list:
                df_meta_event = meta[meta['bg_classname'] == envnt]
                df_meta = pd.concat([df_meta,df_meta_event],axis=0,ignore_index=True)
            df_representation = df_representation.merge(df_meta,how='left',on=['wav_name'],suffixes=('','_'+str('subclass')))
            df_representation = df_representation.reset_index()
            matrix_total, actual_event_total,predict_event_total,wav_name_total= evaluation.evaluationMatrix(df_representation,classification_type,event_list,random_int,opt)


        #matrix_total=pd.concat([matrix_total,matrix],axis=0)

        return matrix_total,wav_name_total,actual_event_total,predict_event_total

    def run_desed(files_per_batch,meta_path,model_event,model_environment,model_m2d,envnt_list,event_list):

        classification_type = opt.classification_type
        audios = list(Path(data_dir).glob('*.wav'))
        df_representation_environ = pd.DataFrame() 
        df_representation_event = pd.DataFrame()
        df_representation_tagging=pd.DataFrame()
        wav_list =[]
        for i in range(files_per_batch):

            random_int = random.randint(0,len(audios)-2)
            audio = audios[random_int:random_int+1]
            files = [*audio]
            print(files)
            #for i in range(len(df_meta)):   
            for fn in files:
                fn_ = str(fn).split('/')[-1]
                print(fn_)
                for target_event in event_list:
                    sr,  target_sr ,lower_db ,higher_db ,k ,hop, class_dir = tagging.config()
                    duration, min_duration = utils.computeDuration(classification_type,pretrain_meta,target_event)
                    top_classes, top_values, secs, sub_representation=tagging.show_topk_sliding_window(class_dir, sr, duration,min_duration,target_sr,lower_db,higher_db,model_m2d, fn, opt, k, hop)
                    #top_classes, top_values, secs, sub_representation=tagging.show_topk_for_all_frames(class_dir, sr, duration,min_duration,target_sr,lower_db,higher_db,model_m2d, fn, opt, k)
                    df_representation_tagging = pd.concat([df_representation_tagging,sub_representation],axis=0,ignore_index=True )
                    wav_list.append(fn_)
            i+=1

        #print('merged original data is: ',df_representation)
        df_representation_tagging.to_csv(output_dir+model_event_name+'_representation_orig_'+timesequence+'_'+str(random_int)+'_'+str(i)+'.csv')
        #df_representation = infer.env_sounds_simple(envnt_list,wav_list,df_representation,model_environment)
        #df_representation = infer.env_sounds(envnt_list,wav_list,df_representation,model_environment)


        
        if classification_type=='event':
            df_representation_event = infer.event_sounds(event_list,df_representation_tagging,model_event)         
            
            if classification_type == 'event':
                df_meta= pd.read_csv(meta_path+'/meta_events.csv')
            elif  classification_type == 'environment':
                df_meta = pd.read_csv(meta_path+'/meta_environments.csv')
            elif  classification_type == 'both':
                df_meta = pd.read_csv(meta_path+'/meta_both.csv')
            df_meta['mixture_audio_filename'] = data_dir+df_meta['wav_name']

            df_representation_event = df_representation_event.merge(df_meta,how='left',on=['wav_name'],suffixes=('','_'+str('subclass')))
            df_representation_event = df_representation_event.reset_index()
            df_representation_event.to_csv(event_dir+model_event_name+'_event_representation_result_'+timesequence+'_'+str(random_int)+'_'+str(i)+'_noisefactor_'+str(noise_factor)+'.csv') 
            matrix_total, actual_event_total,predict_event_total,wav_name_total= evaluation.evaluationMatrix(df_representation_event,classification_type,event_list,random_int,opt)

        elif classification_type=='environment':
            number_of_predicted_environments = infer.config()
            df_representation_environ = infer.env_sounds(envnt_list,wav_list,df_representation_tagging,model_environment,number_of_predicted_environments)

            if classification_type == 'event':
                df_meta= pd.read_csv(meta_path+'/meta_events.csv')
            elif  classification_type == 'environment':
                df_meta = pd.read_csv(meta_path+'/meta_environments.csv')
            elif  classification_type == 'both':
                df_meta = pd.read_csv(meta_path+'/meta_both.csv')
            df_meta['mixture_audio_filename'] = data_dir+df_meta['wav_name']

            df_representation_environ = df_representation_environ.merge(df_meta,how='left',on=['wav_name'],suffixes=('','_'+str('subclass')))
            df_representation_environ = df_representation_environ.reset_index()
            df_representation_environ.to_csv(env_dir+model_environment_name+'_env_representation_result_'+timesequence+'_'+str(random_int)+'_'+str(i)+'_noisefactor_'+str(noise_factor)+'.csv')
            matrix_total, actual_event_total,predict_event_total,wav_name_total= evaluation.evaluationMatrix(df_representation_environ,classification_type,event_list,random_int,opt)

        elif classification_type=='both':
            number_of_predicted_environments = infer.config()
            df_representation_event = infer.event_sounds(event_list,df_representation_tagging,model_event)
            df_representation_event.to_csv(event_dir+model_event_name+'_event_representation_result_'+timesequence+'_'+str(random_int)+'_'+str(i)+'_noisefactor_'+str(noise_factor)+'.csv')
            df_representation_environ = infer.env_sounds(envnt_list,wav_list,df_representation_tagging,model_environment,number_of_predicted_environments)
            df_representation_environ.to_csv(env_dir+model_environment_name+'_env_representation_result_'+timesequence+'_'+str(random_int)+'_'+str(i)+'_noisefactor_'+str(noise_factor)+'.csv')

            
            df_representation = df_representation_event.merge(df_representation_environ,how='outer',on='wav_name')
            
            if classification_type == 'event':
                df_meta= pd.read_csv(meta_path+'/meta_events.csv')
            elif  classification_type == 'environment':
                df_meta = pd.read_csv(meta_path+'/meta_environments.csv')
            elif  classification_type == 'both':
                df_meta = pd.read_csv(meta_path+'/meta_both.csv')
            df_meta['mixture_audio_filename'] = data_dir+df_meta['wav_name']
            
            df_representation = df_representation.merge(df_meta,how='left',on=['wav_name'],suffixes=('','_'+str('subclass')))
            df_representation = df_representation.reset_index()
            df_representation_environ.to_csv(env_dir+model_environment_name+'_both_representation_result_'+timesequence+'_'+str(random_int)+'_'+str(i)+'_noisefactor_'+str(noise_factor)+'.csv')
            matrix_total, actual_event_total,predict_event_total,wav_name_total= evaluation.evaluationMatrix(df_representation,classification_type,event_list,random_int,opt)

        #time.sleep(10)
                    
        return matrix_total,wav_name_total,actual_event_total,predict_event_total

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
    classification_type = opt.classification_type
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

    timesequence=str(start)[-6:]
    print(timesequence)
    print(str(datetime.now()))
    target_events = pd.read_csv('target_events.csv')['target'].values
    target_environments = pd.read_csv('target_environments.csv')['target'].values
    #event_list=['speech','dog', 'cat', 'alarm bell ringing', 'dishes', 'frying', 'blender', 'running water', 'vacuum cleaner', 'electric shaver toothbrush']
    #event_list=['babycry','gunshot', 'glassbreak']
    #event_list=['glassbreak']
    #envnt_list=['beach','bus','cafe/restaurant','car','city_center','forest_path','grocery_store','home','library','metro_station','office','park','residential_area','train','tram']
    #envnt_list=['airport','indoor shopping mall','metro station','pedestrain street','public square','street with medium level of traffic','travelling by a tram','travelling by a bus','travelling by an underground metro','urban park']
    
    classification_type = opt.classification_type
    model_m2d_path = opt.pretrained_m2d
    
    meta_path,model_event_name, model_environment_name,  files_per_batch = experiment.config()
    model_event = SentenceTransformer(model_event_name)
    #model_similarity = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #model_similarity = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    #model_similarity = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    model_environment = SentenceTransformer(model_environment_name)
    pretrain_meta = pd.read_csv('./pretrain_events_meta.csv')
        
    #model_m2d = PortableM2D(weight_file='m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d/weights_ep69it3124-0.47929.pth', num_classes=527)
    model_m2d = PortableM2D(weight_file=model_m2d_path,num_classes=527)
    output=pd.DataFrame()
    for k in range(2):
        matrix_total,wav_name_total,actual_event_total,predict_event_total = experiment.run(files_per_batch,meta_path,model_event,model_environment,model_m2d,target_environments,target_events)
        #matrix_total,wav_name_total,actual_event_total,predict_event_total = experiment.run_desed(files_per_batch,meta_path,model_event,model_environment,model_m2d,target_environments,target_events)
        matrix_total.to_csv(output_dir+model_event_name+'_matrix_total_'+str(k)+'_'+timesequence+'.csv')
        if classification_type!='environment':
            output_ = pd.DataFrame()
            output_['classification_type']=classification_type
            output_['wav_name']=wav_name_total
            output_['actual_event']=actual_event_total
            output_['predict_event']=predict_event_total
            output=pd.concat((output,output_),axis=0)

        time.sleep(10)
        k+=1
    output.to_csv(event_dir+'output_'+timesequence+'.csv')
    end = datetime.now()
    print(start,end)

    with open (log_dir+'env_'+timesequence+'_timecost.log','a') as f:
        f.write('start time is {}\n'.format(str(start)))
        f.write('end time is {}\n'.format(str(end)))   
        f.close()   

    #fig = evaluation.visualize(matrix_total,actual_event_total,predict_event_total,event,'event',opt)
    '''
    for fn in files:
        show_topk_sliding_window(classes, model, fn)

        show_topk_for_all_frames(classes, model, files[0])
    '''