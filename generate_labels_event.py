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
from utils import duration
from portable_m2d import PortableM2D
import pandas as pd
import numpy as np
import random
import gc
import yaml
import csv
import glob
import scipy
from scipy.signal import butter, lfilter, filtfilt,lfilter_zi
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, hamming_loss,jaccard_score,roc_auc_score,roc_curve, precision_recall_curve,auc, f1_score,silhouette_score,normalized_mutual_info_score, adjusted_rand_score
from IPython.display import display, Audio
#import plotnine
#from plotnine import *
import matplotlib.pyplot as plt
import datetime
from datetime import datetime,time
#model_event = SentenceTransformer('stsb-roberta-large')
model_event = SentenceTransformer('all-mpnet-base-v2')
#model_similarity = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#model_similarity = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
#model_similarity = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model_environment = SentenceTransformer('all-mpnet-base-v2')
gc.collect()
classes = pd.read_csv('class_labels_indices.csv').sort_values('mid').reset_index()
#classes[:3]
'''
with zipfile.ZipFile("m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d.zip", "r") as zip_ref:
    zip_ref.extractall(".")
'''

from portable_m2d import PortableM2D
#model = PortableM2D(weight_file='m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d/weights_ep69it3124-0.47929.pth', num_classes=527)
model = PortableM2D(weight_file='m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly/weights_ep67it3124-0.48558.pth',num_classes=527)

noise_factor = 2



class tagging():
    def show_topk_sliding_window(classes, duration,min_duration,target_sr, m2d, wav_file, k=20, hop=1):
        #64000-2,64000-4,16000
        print(wav_file)
        print(m2d.cfg.sample_rate)
        gc.collect()
        wav, sr = librosa.load(wav_file, mono=True, sr=44100)
        #average_wav = np.median(wav)
        #min_wav = min(abs(wav))
        #max_wav = max(abs(wav))
        #noise = np.random.uniform(low=-1.00,high=1.00,size=(len(wav)))*average_wav*noise_factor
        
        #print('type of wav is: ',type(wav))
        #wav = (wav+noise).astype(np.float32)
        #print(wav)
        #wav = float(wav)
        # Loads and shows an audio clip.
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

        #print('type of wav is: ',type(wav))
        # Makes a batch of short segments of the wav into wavs, cropped by the sliding window of [hop, duration].
        wavs = [wav[int(c * sr) : int((c + duration) * sr)] for c in np.arange(0, wav.shape[-1] / sr, hop)]
        wavs = [utils.repeat_if_short(wav,min_duration) for wav in wavs]
        
        wavs = torch.tensor(wavs)
        #print(type(wavs))
        #print(wavs)
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
                

        print()
        
        #print(top_classes)
        wav_name = str(wav_file).split('/')[-1]
        
        sub_representation['time_frame']=secs
        sub_representation['classes']=top_classes
        sub_representation['values']=top_values
        sub_representation['wav_name']=wav_name
        
        sub_representation['time_frame_int']=sub_representation['time_frame'].str.replace('s','')
        sub_representation['time_frame_int']=sub_representation['time_frame_int'].astype(int)
        sub_representation = sub_representation.sort_values(by='time_frame_int').reset_index()
        sub_representation = sub_representation[['wav_name','time_frame','classes','values']]
        
        print(sub_representation)
        
        return top_classes, top_values, secs, sub_representation
    def show_topk_for_all_frames(classes, duration,min_duration,target_sr, m2d, wav_file, k=10):
        print(wav_file)
        print(m2d.cfg.sample_rate)
        # Loads and shows an audio clip.
        wav, sr = librosa.load(wav_file, mono=True, sr=44100)
        wav = librosa.resample(wav, orig_sr=sr, target_sr=32000)
        noise = np.random.uniform(low=-1.0,high=1.0,size=(len(wav)))*wav*noise_factor
        wav = (wav+noise).astype(np.float32)
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

        for i, (probs, ts) in enumerate(zip(probs_per_chunk, timestamps)):
            topk_values, topk_indices = probs.topk(k=k)
            top_classes = [classes.loc[i].display_name for i, v in zip(topk_indices.numpy(), topk_values.numpy())if v>1]
            print('top_classes are: ',top_classes)
            sec = f'{ts/1000:.1f}s '
            print(sec, ', '.join([f'{classes.loc[i].display_name} ({v*100:.1f}%)' for i, v in zip(topk_indices.numpy(), topk_values.numpy()) if v>1]))
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
    def envExtract(df,envnt_list,k=5):
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






class infer():
    def env_sounds_simple(envnt_list,wav_list,df_representation,model_environment):
        print(wav_list)
        gc.collect()
        df_subclass = pd.DataFrame()
        df_subclass['wav_name']=wav_list
        df_subclass['environments_selected']=''
        for i in range(len(df_subclass)):
            gc.collect()
            wav=  df_subclass['wav_name'][i]
            df_wav = df_representation[(df_representation['wav_name']==wav)&(df_representation['classes']!='Silence')]
            print(df_wav)
            sound_wav = df_wav['classes']
            s,v = np.unique(sound_wav,return_counts=True)
            print(s,v)
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
            selected_classes = np.array(df_sound_total['environments_predict'][:8])
            df_subclass['environments_selected'][i]=selected_classes

        print(df_subclass)
        print(len(df_subclass))
        df_representation = df_representation.merge(df_subclass,how='left',on=['wav_name'],suffixes=('','_'+str('subclass')))
        print(df_representation.columns)
        print(df_representation)
        
        
        return df_representation
    def env_sounds(envnt_list,wav_list,df_representation,model_environment
    ):
        print(wav_list)
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
                    print(n,m,class_1,class_2,similarity_class)
                    if (similarity_class>0.5 and similarity_class<1)==True:
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
            print(df_sound)
            df_sound = df_sound[:5]
            
            df_subclass = pd.concat([df_subclass,df_sound],axis=0,ignore_index=True)
        df_subclass = df_subclass.reset_index()
        print(df_subclass)
        print(len(df_subclass))
        df_representation = df_representation.merge(df_subclass,how='left',on=['wav_name','classes'],suffixes=('','_'+str('subclass')))
        #print(df_representation.columns)
        #print(df_representation)

        
        #df_representation = df_representation.drop(['wav_name_subclass','classes_subclass'],axis=1)
        df_representation['environment']=''
        df_representation['similarity_env']=''
        df_representation['environment_top5']=''
        for i in range(len(df_representation)):
            gc.collect()
            if df_representation['rank'][i]>=0:
                label = df_representation['classes_level0'][i]
                scores=[]
                environments=[]
                for environment in envnt_list:
                    label = utils.dataClean(label)
                    environment = utils.dataClean(environment)
                    similarity_score = utils.similarity(label,environment,model_environment)
                    print(i,label,environment,similarity_class)
                    if similarity_score>0.2:
                        environments.append(environment)
                        scores.append(similarity_score)
                    environments_array=np.array(environments)
                    print(environments_array)
                    scores_array=np.array(scores)
                    idx=scores_array.argsort()
                    env_selected =environments_array[idx][:5]
                    print(env_selected)
                        

                    df_representation['similarity_env'][i]=scores
                    df_representation['environment'][i]=environments
                    df_representation['environment_top5'][i]=env_selected


        env_items=[]
        scores=[]
        wavs=[]
        env_items_final=[]
        df_subwav=pd.DataFrame()
        
        for wav in wav_list:
            df_wav_total = df_representation[df_representation['wav_name']==wav]

            df_topk = utils.topk(df_wav_total)
            env_items.append(np.array(df_topk['env_item']))
            environments_extracted_topk=utils.envExtract(df_wav_total,envnt_list)
            env_items_final.append(environments_extracted_topk)
            scores.append(np.array(df_topk['score']))
            wavs.append(wav)
            

        df_subwav['env_selected']=env_items
        df_subwav['score_selected']=scores
        df_subwav['wav_name']=wavs
        df_subwav['env_selected_final']=env_items_final
        df_representation = df_representation.merge(df_subwav,how='left',on=['wav_name'])

        
        return df_representation
    def event_sounds(event_list,df_representation,model_event):
        df_representation['event_predict']=''
        df_representation['similarity']=''
        for i in range(len(df_representation)):
            print(i)
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
class evaluation():
    def evaluationMatrix(df_representation,classification_type,event):
        if classification_type=='event':
            df_event = df_representation[df_representation['event_predict']==event ]
            df_event = df_event.groupby('wav_name')['event_predict'].count().reset_index()
            df_event['event_present_predict'] = True
            df_event= df_event[['wav_name','event_present_predict']]
            df_representation = df_representation.merge(df_event,how='left',on=['wav_name'])
            df_event_result = df_representation[['wav_name','event_present','event_present_predict']]
            df_event_result = df_event_result.drop_duplicates()
            df_event_result = df_event_result.reset_index()
            wav_name_array = df_event_result['wav_name']
            print(df_event_result)
            actual_event = [] 
            predict_event = []
            for i in range(len(df_event_result)):
                print(i,df_event_result['event_present'])
                
                if df_event_result['event_present'][i]==True:
                    actual_event.append(1)
                else:
                    actual_event.append(0)
                if df_event_result['event_present_predict'][i]==True:
                    predict_event.append(1)
                else:
                    predict_event.append(0)
            actual_event=np.array(actual_event)
            predict_event=np.array(predict_event)
        else:
            df_environment = df_representation[['wav_name','env_selected','score_selected','environment_top5','env_level1']]
            df_environment = df_environment.drop_duplicates()
            wav_name_array = df_environment['wav_name']
            accuracy_score_selected=0.0000
            accuracy_score_top5=0.0000
            accuracy_score_level1=0.0000
            total_score = len(df_environment)
            for i in range(len(df_environment)):
                actual_environment=df_environment['bg_classname'][i]
                if actual_environment in df_environment['env_selected'][i]:
                    accuracy_score_selected+=1
                if actual_environment in df_environment['environment_top5'][i]:
                    accuracy_score_top5+=1
                if actual_environment in df_environment['env_level1'][i]:
                    accuracy_score_level1+=1
            accuracy_score_selected = accuracy_score_selected/total_score
            accuracy_score_top5 = accuracy_score_top5/total_score
            accuracy_score_level1 = accuracy_score_level1/total_score



        eventScore=math.sqrt(mean_squared_error(actual_event,predict_event)) 
        print('Train Score: %.5f RMSE' % (eventScore))




        actual_event_2d = actual_event.reshape(-1,1)
        predict_event_2d = predict_event.reshape(-1,1)
        auc_value=roc_auc_score(actual_event_2d,predict_event_2d,average='micro')
        pauc_value=roc_auc_score(actual_event_2d,predict_event_2d,average='micro',max_fpr=0.1)
        cm=confusion_matrix(actual_event,predict_event)
        ari=adjusted_rand_score(actual_event,predict_event)
        nmi=normalized_mutual_info_score(actual_event,predict_event)
        #si=silhouette_score(actual_event_2d,predict_event_2d,metric='euclidean')
        fmeasure=f1_score(actual_event,predict_event,average='micro')
        ac_score=accuracy_score(actual_event,predict_event)
        print('auc: ',auc_value)
        print('pauc: ',pauc_value)
        print('cm_predict: ',cm)
        print('ARI: ',ari)
        print('NMI: ',nmi)
        #print('SI: ',si)
        print('F Measure: ',fmeasure)
        print('Accuracy: ', ac_score)
        mse= mean_squared_error(actual_event,predict_event,multioutput='raw_values')
        print('mse = ',mse)
        
        f= open(project_path+'log/no_noise/'+'mpnet_'+str(random_int)+'_'+classification_type+'_'+event+'.txt','a') 
        f.write('----------------------------------------------------\n')
        f.write('confusion matrix={}\n'.format(cm))
        f.write('auc={}\n'.format(auc_value))
        f.write('pauc={}\n'.format(pauc_value))
        f.write('ARI={}\n'.format(ari))
        f.write('NMI={}\n'.format(nmi))
        #f.write('SI={}\n'.format(si))
        f.write('F Measure={}\n'.format(fmeasure))
        f.write('Accuracy Score={}\n'.format(ac_score))
        f.write('mse={}\n'.format(mse))
        f.close()
                
        matrix=pd.DataFrame()
        matrix['wav_names']=[wav_name_array]
        matrix['random_int']=random_int
        matrix['classification_type']=classification_type
        matrix['event']=event
        matrix['files_per_batch']=files_per_batch
        if classification_type=='environment':
            matrix['accuracy_score_selected']=accuracy_score_selected
            matrix['accuracy_score_top5']=accuracy_score_top5
            matrix['accuracy_score_level1']=accuracy_score_level1
        else:
            matrix['ac_score']=ac_score
            matrix['f_measure']=fmeasure
            matrix['mse']=mse
            matrix['auc']=auc_value
            matrix['pauc']=pauc_value
            matrix['ari']=ari
            matrix['nmi']=nmi
            matrix['confusion_matrix']=[cm]
        

        return matrix, actual_event,predict_event,wav_name_array
    

    def visualize(matrix_total,actual_event_total,predict_event_total):
        x_mean = np.zeros(shape=predict_event_total.shape[0])
        for i in range (len(predict_event_total)):
            x_mean[i] = np.average(a=(predict_event_total[i]))
            i+=1
        act_mean = np.zeros(shape=actual_event_total.shape[0])
        for i in range(actual_event_total.shape[0]):
            act_mean[i] = np.average(a=(actual_event_total[i]))
            i+=1
        
        #plotnine.options.figure_size = (25, 25)
        fig_dist = (ggplot(pd.melt(pd.concat([pd.DataFrame(x_mean, columns=["Predicted Distribution"]).reset_index(drop=True),
                                         pd.DataFrame(act_mean, columns=["Actual Distribution"]).reset_index(drop=True)],
                                        axis=1))) + \
        geom_density(aes(x="value",
                         fill="factor(variable)"), 
                     alpha=0.5,
                     color="black") + \
        geom_point(aes(x="value",
                       y=0,
                       fill="factor(variable)"), 
                   alpha=0.5, 
                   color="black") + \
        labs(title="Distribution of Actual/Predicted "+event )+ \
        xlab("Value") + \
        ylab("Density"))
        #ggsave(plot = fig_dist,filename='output_'+str(object=random_int)+'_'+classification_type+'_'+event+'_distribution.png',dpi=300)
        fig_dist.save(filename='output_'+str(object=random_int)+'_'+classification_type+'_'+event+'_distribution.png',path = '/Users/sharontan6217/Documents/m2d_sed/graph/no_noise',dpi=300)
        #history_dict=loss_history.history
        #history_dict.keys()
        fpr,tpr,thresholds=roc_curve(actual_event_total,predict_event_total)
        roc_auc=auc(fpr,tpr)
        print('roc_auc: ',roc_auc)
        plt.figure(0)
        lw=2
        plt.plot(fpr,tpr,color='orange', lw=lw,label='ROC Curve (area = %0.2f)' % roc_auc)
        plt.plot([0,1],[0,1],color='blue',lw=lw, linestyle='--')
        plt.xlim([0.0000,1.0000])
        plt.ylim([0.0000,1.0500])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curve of Prediction ('+event+')')
        plt.legend(loc='best')
        fig=plt.gcf()
        png_name_roc='output_'+str(object=random_int)+'_'+classification_type+'_'+event+'_roc_curve.png'
        plt.savefig(project_path+'graph/no_noise/'+png_name_roc)
        plt.close()       

        precision,recall,thresholds=precision_recall_curve(actual_event_total,predict_event_total)
        prc_auc=auc(recall,precision)

        
        print('length of precision: ', len(precision),len(recall))
        print('prc_auc: ',prc_auc)
        
        
        
        plt.figure(1)
        plt.plot(recall,precision,color='orange', lw=lw,label='ROC Curve (area = %0.2f)' % prc_auc)
        plt.plot([0,1],[0.5,0.5],color='blue',lw=lw, linestyle='--')
        plt.xlim([0.0000,1.0000])
        plt.ylim([0.0000,1.05000])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve of Prediction: ('+event+')')
        plt.legend(loc='best')
        fig=plt.gcf()
        png_name_roc='output_'+str(object=random_int)+'_'+classification_type+'_'+event+'_prc_curve.png'
        plt.savefig(project_path+'graph/no_noise/'+png_name_roc)
        plt.close()   

        fig,ax=plt.subplots()
        plt.title('Accuracy of Predicted Classification: '+event)
        ax.plot(actual_event_total,color='blue',label='actual')
        ax.set_xlabel('Count')
        ax.set_ylabel('Category(Actual): '+event)
        ax2=ax.twinx()
        ax2.plot(predict_event_total,color='red',label='prediction')
        ax2.set_ylabel('Category(Predict): '+event)
        lines,labels=ax.get_legend_handles_labels()
        lines2,labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines+lines2,labels+labels2,loc=0)
        
        
        fig.set_size_inches(15,7)
        #plt.show()
        png_name_line = 'prediction_line_'+str(object=random_int)+'_'+classification_type+'_'+event+'.png'
        fig.savefig(project_path+'graph/no_noise/'+png_name_line)
        plt.close()

        return fig
if __name__=="__main__":
    start = datetime.now()
    project_path = "./"
    files_per_batch = 5
    meta_path = "/Users/sharontan6217/Documents/m2d_sed/sound_datasets/rare_sound_event/meta"
    output_path = '/Users/sharontan6217/Documents/m2d_sed/event/noise/'
    timesequence=str(start)[-6:]
    print(timesequence)
    print(str(datetime.now()))
    df_meta= utils.metaLoad(meta_path)
    classification_type='event'
    event='gunshot'
    if classification_type=='event':
        if event=='gunshot':
            min_duration =64000
            duration = df_meta[df_meta['event']=='gunshot']['event_length'].max()
            target_sr=32000
        elif event=='glassbreak' :
            min_duration =64000
            duration = df_meta[df_meta['event']=='glassbreak']['event_length'].max()
            target_sr=32000
        elif event=='babycry' :
            min_duration =48000
            duration = df_meta[df_meta['event']=='babycry']['event_length'].max()
            target_sr=16000
        else:
            min_duration = 48000
            duration = 6.
            target_sr=16000
    else:
        min_duration = 48000
        duration = 4.
        target_sr=16000
    #files = np.random.choice(files, size=len(files), replace=False)
    event_list=['babycry','gunshot', 'glassbreak']
    envnt_list=['beach','bus','cafe/restaurant','car','city_center','forest_path','grocery_store','home','library','metro_station','office','park','residential_area','train','tram']
    #random_envnt = random.randrange(0,len(envnt_list_total)-2) 
    #envnt_list=['beach','bus','cafe/restaurant','forest_path']  
    #envnt_list=['train','home','cafe/restaurant','residential_area']   

    df_envnt=pd.DataFrame()
    for envnt in envnt_list:
        df_meta_event = df_meta[df_meta['bg_classname'] == envnt]
        df_envnt = pd.concat([df_envnt,df_meta_event],axis=0,ignore_index=True)
    #print(len(df))
    
    df_envnt = df_envnt.sort_values(by='wav_name',ascending=True)
    df_envnt = df_envnt.reset_index()
    matrix_total=pd.DataFrame()
    actual_event_total=np.array([])
    predict_event_total=np.array([])
    wav_name_total = np.array([])
    #print(df_envnt)
    #print(
    # df_envnt.columns)
    for i in range(1):
        random_int = random.randrange(0,489)
        #random_int = 75
        #files_babycry = list(Path('/Users/sharontan6217/Documents/m2d_sed/sound_datasets/rare_sound_event/eval/audio/babycry').glob('*.wav'))[random_int:random_int+files_per_batch]
        files_gunshot = list(Path('/Users/sharontan6217/Documents/m2d_sed/sound_datasets/rare_sound_event/eval/audio/gunshot').glob('*.wav'))[random_int:random_int+files_per_batch]
        #files_glassbreak = list(Path('/Users/sharontan6217/Documents/m2d_sed/sound_datasets/rare_sound_event/eval/audio/glassbreak').glob('*.wav'))[random_int:random_int+files_per_batch]
        #files = [*files_babycry,*files_glassbreak,*files_gunshot]
        files = [*files_gunshot]
        wav_list =[]
        topk_=[]
        df_representation = pd.DataFrame()
        for j in range(len(df_envnt)):   
            print (j)
            for fn in files:
                fn_ = str(fn).split('/')[-1]
                print(fn_)   
                if fn_ == df_envnt['wav_name'][j]:
                    print(j,fn_,df_envnt['wav_name'][j])

                    top_classes, top_values, secs, sub_representation=tagging.show_topk_sliding_window(classes,duration,min_duration,target_sr,model, fn)

                    df_representation = pd.concat([df_representation,sub_representation],axis=0,ignore_index=True )
                    wav_list.append(fn_)
                    j+=1
        print(df_representation)
        df_representation = df_representation.merge(df_envnt,how='left',on=['wav_name'],suffixes=('','_'+str('subclass')))
        print(df_representation.columns)
        df_representation = df_representation.reset_index()
        #print('merged original data is: ',df_representation)
        df_representation.to_csv(output_path+'representation_orig_'+str(random_int)+'.csv')
        #df_representation = infer.env_sounds_simple(envnt_list,wav_list,df_representation,model_environment)
        #df_representation = infer.env_sounds(envnt_list,wav_list,df_representation,model_environment)
        df_representation = infer.event_sounds(event_list,df_representation,model_event)

        df_representation.to_csv(output_path+'representation_result_'+str(random_int)+'.csv')
        matrix, actual_event,predict_event,wav_name_array= evaluation.evaluationMatrix(df_representation,classification_type,event)
        matrix_total=pd.concat([matrix_total,matrix],axis=0)
        actual_event_total = np.concatenate([actual_event_total,actual_event],axis=0)
        predict_event_total = np.concatenate([predict_event_total,predict_event],axis=0)
        wav_name_total = np.concatenate([wav_name_total,wav_name_array],axis=0)
        i+=1

    
    
    matrix_total.to_csv(output_path+'matrix_total_'+event+'_'+timesequence+'.csv')

    output = pd.DataFrame()
    output['classification_type']=classification_type
    output['wav_name']=wav_name_total
    output['actual_event']=actual_event_total
    output['predict_event']=predict_event_total
    output.to_csv(output_path+'output_'+event+'_'+timesequence+'.csv')
    end = datetime.now()
    print(start,end)
    #fig = evaluation.visualize(matrix_total,actual_event_total,predict_event_total)
    '''
    for fn in files:
        show_topk_sliding_window(classes, model, fn)

        show_topk_for_all_frames(classes, model, files[0])
    '''