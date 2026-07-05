import pandas as pd
import numpy as np
import glob
import scipy
from scipy.signal import butter, lfilter, filtfilt,lfilter_zi
from sentence_transformers import SentenceTransformer, util
import yaml
import math
import numpy as np
import gc

class utils():
    def config():
        with open ('config/tmrl.yaml','r') as f:
            config_duration = yaml.safe_load(f)
        min_duration_babycry = config_duration['data']['min_duration_babycry']
        min_duration_gunshot = config_duration['data']['min_duration_gunshot']
        min_duration_glassbreak = config_duration['data']['min_duration_glassbreak']
        min_duration_others = config_duration['data']['min_duration_others']
        return  min_duration_babycry, min_duration_gunshot, min_duration_glassbreak, min_duration_others
    
    def metaLoad(meta_path):
        print(meta_path)
        
        files = glob.glob('./sound_datasets/rare_sound_event/meta'+'/*.yaml')
        #print(files)

        annotation_string = []
        bg_classname=[]
        bg_path=[]
        ebr=[]
        event_present=[]
        mixture_audio_filename=[]
        event = []
        event_length=[]

        for idx, file in enumerate(files):
            #print(idx,file)
            with open(file) as f:
                data = yaml.safe_load(f)
                #print(data)
                for i in range(len(data)):
                    nested_dict = data[i]
                    #print(nested_dict)
                    if nested_dict['event_present']==False:
                        event_=''
                        event_length_=''
                    else:
                        event_ = nested_dict['annotation_string'].split("_")[2]
                        event_length_=nested_dict['event_length_seconds']
                    event.append(event_)
                    event_length.append(event_length_)
                    annotation_string.append(nested_dict['annotation_string'])
                    bg_classname.append(nested_dict['bg_classname'])
                    bg_path.append(nested_dict['bg_path'])
                    ebr.append(nested_dict['ebr'])
                    event_present.append(nested_dict['event_present'])
                    mixture_audio_filename.append(nested_dict['mixture_audio_filename'])
        df_meta = pd.DataFrame()
        df_meta['mixture_audio_filename']=annotation_string
        df_meta['bg_classname']=bg_classname
        df_meta['bg_path']=bg_path
        df_meta['ebr']=ebr
        df_meta['event_present']=event_present
        df_meta['wav_name']=mixture_audio_filename
        df_meta['event_actual']=event
        df_meta['event_length']=event_length
        df_meta.replace([np.inf,-np.inf],0,inplace=True)
        #print(len(df_meta))
        return df_meta
    def dataClean(text):
        text = str(text).lower()
        text = text.replace('"','')
        text = text.replace("[","")
        text = text.replace("'","")
        text = text.replace("]","")
        return text
    def similarity(sentence1,sentence2,model_similarity):     
        encoding1=model_similarity.encode(sentence1)
        encoding2=model_similarity.encode(sentence2)
        #similarity_score = np.dot(encoding1,encoding2)/(np.linalg.norm(encoding1)*np.linalg.norm(encoding2))
        similarity_score = model_similarity.similarity(encoding1,encoding2)
        return similarity_score
    def computeDuration(classification_type,pretrain_meta,event):
        min_duration_babycry, min_duration_gunshot, min_duration_glassbreak, min_duration_others=utils.config()
        if classification_type=='event' or classification_type=='both':
            if event=='gunshot':
                min_duration = min_duration_gunshot
                duration = np.median(pretrain_meta[pretrain_meta['labels_simple']=='gunshot']['duration'].values)
            elif event=='glassbreak' :
                min_duration = min_duration_glassbreak
                duration = np.median(pretrain_meta[pretrain_meta['labels_simple']=='glass break']['duration'].values)
            elif event=='babycry' :
                min_duration = min_duration_babycry
                duration=np.median(pretrain_meta[pretrain_meta['labels_simple']=='baby cry']['duration'].values)
                #duration = df_meta[df_meta['event_actual']=='babycry']['event_length'].max()
            else:
                min_duration = min_duration_others
                duration = np.median(pretrain_meta[pretrain_meta['labels_simple']=='baby cry']['duration'].values)
        else:
            min_duration = min_duration_others
            duration = np.median(pretrain_meta[pretrain_meta['labels_simple']=='baby cry']['duration'].values)

        return duration, min_duration
    def repeat_if_short(w, min_duration):
        while w.shape[-1] < min_duration:
            w = np.concatenate([w, w], axis=-1)
        return w[..., :min_duration]
    
    def denoise(wavs):
        b,a = butter(2,0.05)
        zi = lfilter_zi(b,a)

        z,_=lfilter(b,a,wavs,zi=zi*wavs[0])
        z2,_=lfilter(b,a,z,zi=zi*z[0])
        wav_denoised = filtfilt(b,a,wavs)

        return wav_denoised
    def topk(df,k=5):
        score=[]
        env=[]
        print(df.columns)
        df=df.drop(['level_0','index'],axis=1)
        df=df.reset_index()
        for i in range(len(df)):
            freq_selected=df['frequency'][i]
            env_item=df['environment'][i]
            similarity_selected=df['similarity_env'][i]
            score_item=[n*freq_selected for n in similarity_selected]
            print(len(env_item),len(score_item))
            for j in range(len(env_item)):
                if env_item[j] not in env:
                    env.append(env_item[j])
                    score.append(score_item[j])
        print(len(env),len(score))
        
        
        df_topk=pd.DataFrame()
        df_topk['env_item']=env
        df_topk['score']=score
        df_topk=df_topk.drop_duplicates()
        df_topk=df_topk.sort_values(by='score',ascending=False)
        df_topk=df_topk[:k]

        return df_topk

    def envFiltering_simple(df,wav,k,n=3):

        #gc.collect()
        score=[]
        env=[]
        wav_names=[]
        print(df.columns)
        #df=df.drop(['index'],axis=1)
        df=df.reset_index()
        #print(df)

        for i in range(len(df)):
            freq_selected=df['frequency'][i]
            env_item=df['environment'][i]
            
            if freq_selected>0:
                print(env_item)
                try:
                    for j in range(len(env_item)):
                        print(i,freq_selected,env_item)
                        if len(env_item[j])>0:
                            env.append(env_item[j])
                            score.append(freq_selected)
                            wav_names.append(wav)
                except TypeError:
                    env_item = [env_item]
                    for j in range(len(env_item)):
                        print(i,freq_selected,env_item)
                        if len(env_item[j])>0:
                            env.append(env_item[j])
                            score.append(freq_selected)
                            wav_names.append(wav)

        #print(env)
        #print(score)           
        df_filtering_simple = pd.DataFrame()
        df_filtering_simple['env_items']=env
        df_filtering_simple['scores']=score
        df_filtering_simple['wav_name']=wav_names
        df_filtering_simple=df_filtering_simple.drop_duplicates()
        #print(df_filtering_simple)
        df_filtering_simple_transform = df_filtering_simple.groupby('env_items')['scores'].sum().reset_index()
        #print(df_filtering_simple_transform)
        df_filtering_simple.to_csv('df_filtering_simple_'+str(wav)+'.csv')
        df_filtering_simple_transform = df_filtering_simple_transform.sort_values(by='scores',ascending=False)
        df_score=utils.rank(df_filtering_simple_transform['scores'],wav)
        df_filtering_simple_transform = df_filtering_simple_transform.merge(df_score,how='left',on=['scores'])
        rank_simple =df_filtering_simple_transform['scores'].rank(ascending=False)
        df_filtering_simple_transform['rank']=rank_simple
        df_filtering_simple_transform.to_csv('df_filtering_simple_trasnform_'+str(wav)+'.csv')
        if min(df_filtering_simple_transform['rank'])<=k:
            df_filtering_simple_transform=df_filtering_simple_transform[df_filtering_simple_transform['rank']<=k]
        else:
            df_filtering_simple_transform=df_filtering_simple_transform[df_filtering_simple_transform['rank_abs']<n]
        return df_filtering_simple_transform
    def envFiltering_weighted(df,wav,k):
        #gc.collect()
        score=[]
        env=[]
        wav_names=[]
        print(df.columns)
        #df=df.drop(['index'],axis=1)
        
        df=df.reset_index()
        for i in range(len(df)):
            freq_selected=df['frequency'][i]
            env_item=df['environment'][i]
            similarity_env = df['similarity_env'][i]
            
            if freq_selected>0:
                for j in range(len(env_item)):
                    print(i,freq_selected,env_item)
                    if len(env_item[j])>0:
                        env.append(env_item[j])
                        score.append(freq_selected*similarity_env[j])
                        wav_names.append(wav)
        #print(env)
        #print(score)           
        df_filtering_weighted = pd.DataFrame()
        df_filtering_weighted['env_items']=env
        df_filtering_weighted['scores']=score
        df_filtering_weighted['scores']=df_filtering_weighted['scores'].astype(float)
        df_filtering_weighted['wav_name']=wav_names
        df_filtering_weighted=df_filtering_weighted.drop_duplicates()
        #print(df_filtering_weighted)
        df_filtering_weighted_transform = df_filtering_weighted.groupby('env_items')['scores'].sum().reset_index()
        df_filtering_weighted_transform['scores']=df_filtering_weighted_transform['scores'].astype(float)
        #print(df_filtering_weighted_transform)
        df_filtering_weighted.to_csv('df_filtering_weighted_'+str(wav)+'.csv')
        df_filtering_weighted_transform = df_filtering_weighted_transform.sort_values(by='scores',ascending=False)
        rank_weighted =df_filtering_weighted_transform['scores'].rank(ascending=False)
        df_filtering_weighted_transform['rank']=rank_weighted
        df_filtering_weighted_transform.to_csv('df_filtering_weighted_trasnform_'+str(wav)+'.csv')
        df_filtering_weighted_transform=df_filtering_weighted_transform[df_filtering_weighted_transform['rank']<=k]
        return df_filtering_weighted_transform
    def envExtract(df,wav,model_similarity_envrn,envnt_list,k):
        #ipca clustering and define
        print(df.columns)
        #df=df.drop(['index'],axis=1)
        df=df.reset_index()
        env_extracted=[]
        score_extracted=[]
        classes_=[]
        class_extracted=[]
        for i in range(len(df)):
            if len(df['environment'][i])>0:
                if len(classes_)==0:
                    classes_=df['environment'][i]
                else:
                    classes_=df['environment'][i]+classes_
        for env in envnt_list:
            classes_ = utils.dataClean(classes_)
            env = utils.dataClean(env)
            similarity_env=utils.similarity(classes_,env,model_similarity_envrn)
            #print(len(classes_),len(env))
            if similarity_env>0.1:
                env_extracted.append(env)
                score_extracted.append(similarity_env)
                class_extracted.append(classes_)
            #print(env_extracted,score_extracted)
            df_level1 = pd.DataFrame()
            df_level1['env_items']=env_extracted
            df_level1['scores']=score_extracted            
            df_level1['classes']=class_extracted
            df_level1['wav_name']=wav
            df_level1=df_level1.drop_duplicates()
            df_level1 = df_level1.sort_values(by='scores',ascending=False)
            rank_weighted =df_level1['scores'].rank(ascending=False)
            df_level1['rank']=rank_weighted
            df_level1.to_csv('df_level1_trasnform_'+str(wav)+'.csv')
            df_level1=df_level1[df_level1['rank']<=k]
            return df_level1
    def rank(score_total,wav):
        df_score=pd.DataFrame()
        df_score['scores']=score_total
        df_score['scores']=df_score['scores'].astype(float)
        df_score=df_score.drop_duplicates()
        df_score['rank_abs']=df_score['scores'].rank(ascending=False)
        df_score.to_csv('score_'+wav+'.csv')
        return df_score
    def ci(x,accuracy,z):
        n = len(x)-1
        std = math.sqrt((accuracy*(1-accuracy))/n)
        ci = [accuracy-z*std,accuracy+z*std]
        return ci





