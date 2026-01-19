import yaml
import csv
import glob
import pandas as pd
import numpy as np
files = glob.glob('./*.yaml')

annotation_string = []
bg_classname=[]
bg_path=[]
ebr=[]
event_present=[]
mixture_audio_filename=[]
event = []
event_length=[]

for idx, file in enumerate(files):
    print(idx,file)
    with open(file) as f:
        data = yaml.safe_load(f)
        #print(data)
        for i in range(len(data)):
            nested_dict = data[i]
            print(nested_dict)
            if nested_dict['event_present']==False:
                event_=''
                event_length_=''
            else:
                event_ = nested_dict['annotation_string'].split("_")[2]
                event_length_=nested_dict['event_length_seconds']
            print(event_)
            event.append(event_)
            event_length.append(event_length_)

            annotation_string.append(nested_dict['annotation_string'])
            bg_classname.append(nested_dict['bg_classname'])
            bg_path.append(nested_dict['bg_path'])
            ebr.append(nested_dict['ebr'])
            event_present.append(nested_dict['event_present'])
            mixture_audio_filename.append(nested_dict['mixture_audio_filename'])
df = pd.DataFrame()
df['mixture_audio_filename']=annotation_string
df['bg_classname']=bg_classname
df['bg_path']=bg_path
df['ebr']=ebr
df['event_present']=event_present
df['mixture_audio_filename']=mixture_audio_filename
df['event']=event
df['event_length']=event_length
df.replace([np.inf,-np.inf],0,inplace=True)
print(len(df))
df.to_csv('meta.csv')
print(df[df['event']=='glassbreak']['event_length'].max())
 