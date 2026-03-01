import os
import warnings; warnings.simplefilter('ignore')
import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import zipfile
import librosa
#import utils
from portable_m2d import PortableM2D
import pandas as pd
import numpy as np
import random
import gc
import yaml
import csv
import glob
from sentence_transformers import SentenceTransformer, util
from IPython.display import display, Audio
gc.collect()
#model_similarity_event = SentenceTransformer('stsb-roberta-large')
#model_similarity_event = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model_similarity_event = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
#model_similarity_event = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#model_similarity_event = SentenceTransformer('all-mpnet-base-v2')
model_similarity_envrn = SentenceTransformer('stsb-roberta-large')
#model_similarity_envrn = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
#model_similarity_envrn = SentenceTransformer('all-mpnet-base-v2')
classes = pd.read_csv('class_labels_indices.csv').sort_values('mid').reset_index()
#classes[:3]
'''
with zipfile.ZipFile("m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d.zip", "r") as zip_ref:
    zip_ref.extractall(".")
'''

from portable_m2d import PortableM2D
#model = PortableM2D(weight_file='m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d/weights_ep69it3124-0.47929.pth', num_classes=527)
model = PortableM2D(weight_file='m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly/weights_ep67it3124-0.48558.pth', num_classes=527)

class tagging():
    def show_topk_sliding_window(classes, m2d, wav_file, k=20, hop=1):
        #64000-2,64000-4,16000
        print(wav_file)
        print(m2d.cfg.sample_rate)

        gc.collect()
        wav, sr = librosa.load(wav_file, mono=True, sr=44100)
        if 'gunshot' in str(wav_file) or 'glassbreak' in str(wav_file) :
            min_duration =48000 
            duration = 2.
            target_sr=16000       
        elif 'babycry' in str(wav_file):
            min_duration =48000
            duration = 4.
            target_sr=16000
        else:
            min_duration = 64000
            duration = 6.
            target_sr=16000
        # Loads and shows an audio clip.
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        # Makes a batch of short segments of the wav into wavs, cropped by the sliding window of [hop, duration].
        wavs = [wav[int(c * sr) : int((c + duration) * sr)] for c in np.arange(0, wav.shape[-1] / sr, hop)]
        wavs = [utils.repeat_if_short(wav,min_duration) for wav in wavs]
        wavs = torch.tensor(wavs)
        #wav_ = str(os.path.dirname(wav_file))+"/"+str(os.path.basename(wav_file))
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
        wav_name = str(wav_file).split('\\')[-1]
        
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
    def show_topk_for_all_frames(classes, m2d, wav_file, k=10):
        print(wav_file)
        print(m2d.cfg.sample_rate)
        # Loads and shows an audio clip.
        wav, sr = librosa.load(wav_file, mono=True, sr=44100)
        wav = librosa.resample(wav, orig_sr=sr, target_sr=32000)
        display(Audio(wav, rate=m2d.cfg.sample_rate))
        wav = torch.tensor(wav)
        #wav_ = str(os.path.dirname(wav_file))+"/"+str(os.path.basename(wav_file))
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
    def metaLoad(meta_dir):
        files = glob.glob(meta_dir+'/*.yaml')
        #print(files)

        annotation_string = []
        bg_classname=[]
        bg_path=[]
        ebr=[]
        event_present=[]
        mixture_audio_filename=[]
        event = []


        for idx, file in enumerate(files):
            #print(idx,file)
            with open(file) as f:
                data = yaml.safe_load(f)
                #print(data)
                for i in range(len(data)):
                    nested_dict = data[i]
                    #print(nested_dict)
                    event_ = nested_dict['annotation_string'].split("_")[2]
                    #print(event_)
                    if nested_dict['event_present']==False:
                        event_=''
                    event.append(event_)

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
    def rank(score_total,wav):
        df_score=pd.DataFrame()
        df_score['scores']=score_total
        df_score['scores']=df_score['scores'].astype(float)
        df_score=df_score.drop_duplicates()
        df_score['rank_abs']=df_score['scores'].rank(ascending=False)
        df_score.to_csv('score_'+wav+'.csv')
        return df_score

    def envFiltering_simple(df,wav,k1=5,k2=3):
        score=[]
        env=[]
        wav_names=[]
        print(df.columns)
        df=df.drop(['level_0','index'],axis=1)
        df=df.reset_index()
        #print(df)
        for i in range(len(df)):
            freq_selected=df['frequency'][i]
            env_item=df['environment'][i]
            
            if freq_selected>0:
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
        if min(df_filtering_simple_transform['rank'])<=k1:
            df_filtering_simple_transform=df_filtering_simple_transform[df_filtering_simple_transform['rank']<=k1]
        else:
            df_filtering_simple_transform=df_filtering_simple_transform[df_filtering_simple_transform['rank_abs']<=k2]
        return df_filtering_simple_transform
    def envFiltering_weighted(df,wav,k=6):
        score=[]
        env=[]
        wav_names=[]
        print(df.columns)
        df=df.drop(['level_0','index'],axis=1)
        
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
        df_filtering_weighted_transform=df_filtering_weighted_transform[df_filtering_weighted_transform['rank']<k]
        return df_filtering_weighted_transform
    def envExtract(df,wav,envnt_list,k=5):
        #ipca clustering and define
        print(df.columns)
        df=df.drop(['level_0','index'],axis=1)
        df=df.reset_index()
        env_extracted=[]
        score_extracted=[]
        classes_=[]
        class_extracted=[]
        for i in range(len(df)):
            if len(df['environment'][i])>0:
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






class infer():
    def env_sounds_simple(envnt_list,wav_list,df_representation,model_similarity_envrn):
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
                similarity_total_ = utils.similarity(s,env,model_similarity_envrn)
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
    def env_sounds(envnt_list,wav,df_representation,model_similarity_envrn    ):
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
                similarity_class = utils.similarity(class_1,class_2,model_similarity_envrn)
                print(n,m,class_1,class_2,similarity_class)
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
        print(df_sound)
        df_sound = df_sound[:5]
            
        df_sound = df_sound.reset_index()
        print(df_sound)
        print(len(df_sound))
        df_representation = df_representation.merge(df_sound,how='left',on=['wav_name','classes'],suffixes=('','_'+str('subclass')))
        #print(df_representation.columns)
        #print(df_representation)

        
        #df_representation = df_representation.drop(['wav_name_subclass','classes_subclass'],axis=1)
        df_representation['environment']=''
        df_representation['similarity_env']=''
        for i in range(len(df_representation)):
            gc.collect()
            if df_representation['rank'][i]>=0:
                label = df_representation['classes_level0'][i]
                scores=[]
                environments=[]
                for environment in envnt_list:
                    label = utils.dataClean(label)
                    environment = utils.dataClean(environment)
                    similarity_score = utils.similarity(label,environment,model_similarity_envrn)
                    print(i,label,environment,similarity_class)
                    if similarity_score>0.1:
                        environments.append(environment)
                        scores.append(similarity_score)
                    environments_array=np.array(environments)
                    print(environments_array)
                    df_representation['similarity_env'][i]=scores
                    df_representation['environment'][i]=environments
        env_top5=[]
        env_items=[]
        scores=[]
        wavs=[]
        env_items_level1=[]
        df_subwav=pd.DataFrame()
        
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

        df_subwav['env_selected']=df_filtering_weighted_transform['env_items']
        df_subwav['score_selected']=df_filtering_weighted_transform['scores']
        df_subwav['environment_top5']=df_filtering_simple_transform['env_items']
        df_subwav['env_level1']=df_level1['env_items']
        df_subwav['wav_name']=wav
        df_representation = df_representation.merge(df_subwav,how='left',on=['wav_name'])

        
        return df_representation
    def event_sounds(event_list,df_representation,model_similarity_event):
        df_representation['event_predict']=''
        df_representation['similarity']=''
        for i in range(len(df_representation)):
            print(i)
            gc.collect()
            
            label = df_representation['classes'][i]
            for event in event_list:
                label = utils.dataClean(label)
                event = utils.dataClean(event)
                similarity_score = utils.similarity(label,event,model_similarity_event)
                print(i,label,event,similarity_score)
                if similarity_score>0.4:
                    df_representation['event_predict'][i]=event
                    df_representation['similarity'][i]=similarity_score
        return df_representation
class evaluation():
    def evaluationMatrix():
        return matrix
    def visualize():
        return fig
if __name__=="__main__":
    project_path = "../"
    meta_dir = "C:/Users/sharo/Documents/GitHub/TMRL/sound_datasets/rare_sound_event/meta"
    event_list=['baby cry','gun shot', 'glass break']
    envnt_list=['beach','bus','cafe/restaurant','car','city_center','forest_path','grocery_store','home','library','metro_station','office','park','residential_area','train','tram']
    for i in range(0,1):
        #random_int = random.randrange(0,480)
        random_int = 1
        files_babycry = list(Path('C:/Users/sharo/Documents/GitHub/TMRL/sound_datasets/rare_sound_event/eval/audio/babycry').glob('*.wav'))[random_int]
        files_glassbreak = list(Path('C:/Users/sharo/Documents/GitHub/TMRL/sound_datasets/rare_sound_event/eval/audio/glassbreak').glob('*.wav'))[random_int]
        files_gunshot = list(Path('C:/Users/sharo/Documents/GitHub/TMRL/sound_datasets/rare_sound_event/eval/audio/gunshot').glob('*.wav'))[random_int]
        files = [files_babycry,files_glassbreak,files_gunshot]
        #files = np.random.choice(files, size=len(files), replace=False)

        #random_envnt = random.randrange(0,len(envnt_list_total)-2) 
        #envnt_list=['beach','bus','cafe/restaurant','forest_path']  
        #envnt_list=['train','home','cafe/restaurant','residential_area']   
        df_meta= utils.metaLoad(meta_dir)
        df_envnt=pd.DataFrame()
        for envnt in envnt_list:
            df_meta_event = df_meta[df_meta['bg_classname'] == envnt]
            df_envnt = pd.concat([df_envnt,df_meta_event],axis=0,ignore_index=True)
        #print(len(df))
        
        df_envnt = df_envnt.sort_values(by='wav_name',ascending=True)
        df_envnt = df_envnt.reset_index()
        #print(df_envnt)
        #print(df_envnt.columns)

        topk_=[]
        df_representation_total = pd.DataFrame()
        df_representation_orig = pd.DataFrame()
        for i in range(len(df_envnt)):   
            for fn in files:
                fn_ = str(fn).split('\\')[-1]
                print(fn_)   
                if fn_ == df_envnt['wav_name'][i]:
                    print(i,fn_,df_envnt['wav_name'][i])

                    top_classes, top_values, secs, sub_representation=tagging.show_topk_sliding_window(classes, model, fn)
                    df_representation = sub_representation.merge(df_envnt,how='left',on=['wav_name'],suffixes=('','_'+str('subclass')))
                    print(df_representation.columns)
                    df_representation = df_representation.reset_index()
                    #print('merged original data is: ',df_representation)
                    df_representation_orig =  pd.concat([df_representation_orig,df_representation],axis=0,ignore_index=True )
                    #df_representation = infer.env_sounds_simple(envnt_list,wav_list,df_representation,model_similarity_envrn )
                    #df_representation = infer.env_sounds(envnt_list,fn_,df_representation,model_similarity_envrn )
                    df_representation = infer.event_sounds(event_list,df_representation,model_similarity_event)
                    df_representation_total = pd.concat([df_representation_total,df_representation],axis=0,ignore_index=True )

        #print(df_representation)
        df_representation_orig.to_csv('representation_orig_'+str(random_int)+'_roberta_glassbreak_48000_2_gunshot_48000_16000.csv')
        df_representation_total.to_csv('representation_result_'+str(random_int)+'_roberta_glassbreak_48000_2_gunshot_48000_16000.csv')
        '''
        for fn in files:
            show_topk_sliding_window(classes, model, fn)

            show_topk_for_all_frames(classes, model, files[0])
        '''
        i+=1