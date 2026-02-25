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
import utils
from portable_m2d import PortableM2D
import pandas as pd
import numpy as np
import random
import gc

from scipy import fft
from scipy.io import wavfile
from scipy.stats import spearmanr,pearsonr
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, hamming_loss,jaccard_score,roc_auc_score,roc_curve, precision_recall_curve,auc, f1_score,silhouette_score,normalized_mutual_info_score, adjusted_rand_score
from IPython.display import display, Audio

import datetime
from datetime import datetime
import utils
from utils import utils

gc.collect()




class tagging():
    def show_topk_sliding_window(classes,  duration,min_duration,target_sr,m2d, wav_file,opt, k=20, hop=1):
        #64000-2,64000-4,16000
        #print(wav_file)
        noise_factor = opt.noise_factor
        print(m2d.cfg.sample_rate)
        gc.collect()
        wav, sr = librosa.load(wav_file, mono=True, sr=44100)      
        average_wav = np.median(wav)  
        noise = np.random.uniform(low=-1.0, high=1.0, size=len(wav))*average_wav*noise_factor
        #print(wav)
        #print(noise.shape)
        wav = (wav+noise).astype(np.float32)


        # Loads and shows an audio clip.
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        # Makes a batch of short segments of the wav into wavs, cropped by the sliding window of [hop, duration].
        wavs = [wav[int(c * sr) : int((c + duration) * sr)] for c in np.arange(0, wav.shape[-1] / sr, hop)]
        wavs = [utils.repeat_if_short(wav,min_duration) for wav in wavs]

        wavs = torch.tensor(wavs)
        wav_ = str(os.path.dirname(wav_file))+"/"+str(os.path.basename(wav_file))
        # Predicts class probabilities for the batch segments.
        with torch.no_grad():
            try:
                probs_per_chunk = m2d(wavs).softmax(1)
            except Exception as error:
                print(error)
                with open ("error.log", "a") as f:
                    f.write(str(error))
                    f.close()
                probs = []
        # Shows the top-k prediction results.
        top_classes=[]
        top_values=[]
        secs=[]
        sub_representation = pd.DataFrame()
        for i, probs in enumerate(probs_per_chunk):
            topk_values, topk_indices = probs.topk(k=k)
            sec = f'{i * hop:d}s '
            print(sec, ', '.join([f'{classes.loc[i].display_name} ({v*100:.1f}%)' for i, v in zip(topk_indices.numpy(), topk_values.numpy())]))
            for i, v in zip(topk_indices.numpy(), topk_values.numpy()):
                top_classes.append(classes.loc[i].display_name )
                top_values.append(v)
                secs.append(sec)
                

        
        #print(top_classes)
        wav_name = str(wav_file).split('\\')[-1]
        
        sub_representation['time_frame']=secs
        sub_representation['classes']=top_classes
        sub_representation['values']=top_values
        sub_representation['wav_name']=wav_name
        
        sub_representation['time_frame_int']=sub_representation['time_frame'].str.replace('s','')
        sub_representation['time_frame_int']=sub_representation['time_frame_int'].astype(int)
        sub_representation = sub_representation.sort_values(by='time_frame_int').reset_index()
        sub_representation = sub_representation[['wav_name','time_frame','classes','values']]
        
        #print(sub_representation)
        
        return top_classes, top_values, secs, sub_representation
    
    def show_topk_for_all_frames(classes,  duration,min_duration,target_sr,m2d, wav_file, opt, k=20):
        #print(wav_file)
        print(m2d.cfg.sample_rate)
        # Loads and shows an audio clip.
        wav, sr = librosa.load(wav_file, mono=True, sr=44100)      
        noise_factor = opt.noise_factor
        print(m2d.cfg.sample_rate)
        average_wav = np.median(wav)  
        noise = np.random.uniform(low=-1.0, high=1.0, size=len(wav))*average_wav*noise_factor
        #print(wav)
        #print(noise.shape)
        wav = (wav+noise).astype(np.float32)
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        display(Audio(wav, rate=m2d.cfg.sample_rate))
        wav = torch.tensor(wav)
        wav_ = str(os.path.dirname(wav_file))+"/"+str(os.path.basename(wav_file))
        # Predicts class probabilities for all frames.
        with torch.no_grad():
            logits_per_chunk, timestamps = m2d.forward_frames(wav.unsqueeze(0))  # logits_per_chunk: [1, 62, 527], timestamps: [1, 62]
            probs_per_chunk = logits_per_chunk.squeeze(0).softmax(-1)  # logits [1, 62, 527] -> probabilities [62, 527]
            timestamps = timestamps[0]  # [1, 62] -> [62]
        # Shows the top-k prediction results.
        time_frame=[]
        top_classes=[]
        top_values=[]
        secs=[]
        sub_representation = pd.DataFrame()
        for i, (probs, ts) in enumerate(zip(probs_per_chunk, timestamps)):
            topk_values, topk_indices = probs.topk(k=k)
            top_classes = [classes.loc[i].display_name for i, v in zip(topk_indices.numpy(), topk_values.numpy())if v>1]
            print('top_classes are: ',top_classes)
            sec = f'{ts/1000:.1f}s '
            print(sec, ', '.join([f'{classes.loc[i].display_name} ({v*100:.1f}%)' for i, v in zip(topk_indices.numpy(), topk_values.numpy()) if v>1]))
            for i, v in zip(topk_indices.numpy(), topk_values.numpy()):
                top_classes.append(classes.loc[i].display_name )
                top_values.append(v)
                secs.append(sec)
                

        
        #print(top_classes)
        wav_name = str(wav_file).split('\\')[-1]
        
        sub_representation['time_frame']=secs
        sub_representation['classes']=top_classes
        sub_representation['values']=top_values
        sub_representation['wav_name']=wav_name
        
        sub_representation['time_frame_int']=sub_representation['time_frame'].str.replace('s','')
        sub_representation['time_frame_int']=sub_representation['time_frame_int'].astype(int)
        sub_representation = sub_representation.sort_values(by='time_frame_int').reset_index()
        sub_representation = sub_representation[['wav_name','time_frame','classes','values']]
        
        #print(sub_representation)
        
        return top_classes, top_values, secs, sub_representation
class infer():
    def env_sounds_simple(envnt_list,wav_list,df_representation,model_environment,k=5):
        #print(wav_list)
        gc.collect()
        df_subclass = pd.DataFrame()
        df_subclass['wav_name']=wav_list
        df_subclass['environments_selected']=''
        for i in range(len(df_subclass)):
            wav=  df_subclass['wav_name'][i]
            df_wav = df_representation[(df_representation['wav_name']==wav)&(df_representation['classes']!='Silence')]
            #print(df_wav)
            sound_wav = df_wav['classes']
            s,v = np.unique(sound_wav,return_counts=True)
            #print(s,v)
            classes=[]
            similarity_total=[]
            for env in envnt_list:
                s = utils.dataClean(s)
                similarity_total_ = utils.similarity(s,env,model_environment)
                print(s,env,similarity_total_)
                classes.append(env)
                similarity_total.append(similarity_total_)
            df_sound_total=pd.DataFrame()
            df_sound_total['environments_predict']=classes
            df_sound_total['similarity_simple']=similarity_total
            df_sound_total = df_sound_total.sort_values(by='similarity_simple',ascending=False)
            selected_classes = np.array(df_sound_total['environments_predict'][:k])
            df_subclass['environments_selected'][i]=selected_classes

        df_representation = df_representation.merge(df_subclass,how='left',on=['wav_name'],suffixes=('','_'+str('subclass')))

        
        
        return df_representation
    def env_sounds(envnt_list,wav_list,df_representation,model_environment,k=5    ):

        gc.collect()
        #print(wav_list)
        df_subclass = pd.DataFrame()
        for wav in wav_list:
            df_wav = df_representation[(df_representation['wav_name']==wav)&(df_representation['classes']!='Silence')]
            #print(df_wav)
            sound_wav = df_wav['classes']
            s,v = np.unique(sound_wav,return_counts=True)
            #print(s,v)
            df_sound = pd.DataFrame()
            df_sound['classes']=s
            df_sound['frequency']=v
            df_sound['classes'] = df_sound['classes'].astype(str)
            df_sound['frequency'] = df_sound['frequency'].astype(float)
            df_sound['similarity_classes']=''
            df_sound['classes_level0']=''
            for n in range(len(df_sound)-1):
                similarity_classes=[]
                classes_level0=[]
                class_1 = df_sound['classes'][n]
                classes_level0= []
                classes_level0.append(class_1)
                frequency_1 = df_sound['frequency'][n]
                for m in range(n+1,len(df_sound)):
                    frequency_2 = df_sound['frequency'][m]
                    class_2 = df_sound['classes'][m]                
                    similarity_class = utils.similarity(class_1,class_2,model_environment)
                    #print(n,m,class_1,class_2,similarity_class)
                    if (similarity_class>=0.6 and similarity_class<1)==True:
                        print(n,class_1,class_2,similarity_class)
                        similarity_classes.append(similarity_class)
                        classes_level0.append(class_2)
                        frequency_1=frequency_1+frequency_2
                        print('frequency is updated to ',frequency_1)
                #print('frequency_1 is: ',frequency_1)   
                df_sound['frequency'][n] = frequency_1            
                df_sound['classes_level0'][n] = classes_level0
                df_sound['similarity_classes'][n]=  similarity_classes

            rank = df_sound['frequency'].rank(ascending=False)
            df_sound['rank']=rank
            df_sound['wav_name']=wav
            df_sound = df_sound.sort_values(by='frequency',ascending=False)
            #print(df_sound)
            df_sound = df_sound[:k]
            
            df_subclass = pd.concat([df_subclass,df_sound],axis=0,ignore_index=True)
        df_subclass = df_subclass.reset_index()

        df_representation = df_representation.merge(df_subclass,how='left',on=['wav_name','classes'],suffixes=('','_'+str('subclass')))


        
        #df_representation = df_representation.drop(['wav_name_subclass','classes_subclass'],axis=1)
        df_representation['environment']=''
        df_representation['similarity_env']=''
        for i in range(len(df_representation)):
            
            if df_representation['rank'][i]>=0:
                label = df_representation['classes_level0'][i]
                scores=[]
                environments=[]
                for environment in envnt_list:
                    label = utils.dataClean(label)
                    environment = utils.dataClean(environment)
                    similarity_score = utils.similarity(label,environment,model_environment)
                    print(i,label,environment,similarity_score)
                    if similarity_score>0.1:
                        environments.append(environment)
                        scores.append(similarity_score)
                    environments_array=np.array(environments)
                    #print(environments_array)
                    df_representation['similarity_env'][i]=scores
                    df_representation['environment'][i]=environments

        env_top5=[]
        env_items=[]
        scores=[]
        wavs=[]
        env_items_level1=[]
        df_subwav=pd.DataFrame()
        
        for wav in wav_list:
            df_wav_total = df_representation[df_representation['wav_name']==wav]
            df_filtering_simple_transform = utils.envFiltering_simple(df_wav_total,wav)
            df_filtering_weighted_transform = utils.envFiltering_weighted(df_wav_total,wav)
            df_level1=utils.envExtract(df_wav_total,wav,envnt_list)
            env_top5.append(np.array(df_filtering_simple_transform['env_items']))
            env_items.append(np.array(df_filtering_weighted_transform['env_items']))
            env_items_level1.append(np.array(df_level1['env_items']))
            scores.append(np.array(df_filtering_weighted_transform['scores']))
            wavs.append(wav)
            

        df_subwav['env_selected']=env_items
        df_subwav['score_selected']=scores
        df_subwav['wav_name']=wavs
        df_subwav['environment_top5']=env_top5
        df_subwav['env_level1']=env_items_level1
        df_representation = df_representation.merge(df_subwav,how='left',on=['wav_name'])

        
        return df_representation
    def event_sounds(event_list,df_representation,model_event):
        df_representation['event_predict']=''
        df_representation['similarity']=''
        for i in range(len(df_representation)):
            gc.collect()
            
            label = df_representation['classes'][i]
            for event in event_list:
                label = utils.dataClean(label)
                event = utils.dataClean(event)
                similarity_score = utils.similarity(label,event,model_event)
                print(i,label,event,similarity_score)
                if similarity_score>=0.5:
                    df_representation['event_predict'][i]=event
                    df_representation['similarity']=similarity_score
        return df_representation