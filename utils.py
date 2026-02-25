import pandas as pd
import numpy as np
import glob
import scipy
from scipy.signal import butter, lfilter, filtfilt,lfilter_zi
from sentence_transformers import SentenceTransformer, util
import yaml

class utils():
    def metaLoad(meta_path):
        files = glob.glob(meta_path+'/*.yaml')
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
        similarity_score = np.dot(encoding1,encoding2)/(np.linalg.norm(encoding1)*np.linalg.norm(encoding2))
        return similarity_score
    def computeDuration(classification_type,df_meta,event):

        if classification_type=='event' or classification_type=='both':
            if event=='gunshot':
                min_duration =64000
                duration = df_meta[df_meta['event_actual']=='gunshot']['event_length'].max()
                target_sr=32000
            elif event=='glassbreak' :
                min_duration =64000
                duration = df_meta[df_meta['event_actual']=='glassbreak']['event_length'].max()
                target_sr=32000
            elif event=='babycry' :
                min_duration =48000
                duration = df_meta[df_meta['event_actual']=='babycry']['event_length'].max()
                target_sr=16000
            else:
                min_duration = 48000
                duration = 6.
                target_sr=16000
        else:
            min_duration = 48000
            duration = 4.
            target_sr=16000
        return min_duration,duration,target_sr
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
    def envExtract(df,envnt_list,model_environment,k=5):
        #ipca clustering and define
        print(df.columns)
        df=df.drop(['level_0','index'],axis=1)
        df=df.reset_index()
        env_extracted=[]
        score_extracted=[]
        class_final=[]
        for i in range(len(df)):
            if len(df['environment'][i])>0:
                class_final=df['environment'][i]+class_final
        for env in envnt_list:
            class_final = utils.dataClean(class_final)
            env = utils.dataClean(env)
            similarity_env=utils.similarity(class_final,env,model_environment)
            print(len(class_final),len(env))
            if similarity_env>0.2:
                env_extracted.append(env)
                score_extracted.append(similarity_env)
            environments_extracted=np.array(env_extracted)
            print(env_extracted,score_extracted)
            scores_extracted=np.array(score_extracted)
            idx=scores_extracted.argsort()
            environments_extracted_topk =environments_extracted[idx][:k]
            print(environments_extracted_topk)
            with open ('final.log','a') as f:
                f.write(str(score_extracted))
                f.write(str(environments_extracted))
            f.close()

        return environments_extracted_topk



