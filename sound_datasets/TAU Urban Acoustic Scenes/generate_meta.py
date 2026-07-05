import pandas as pd

total = pd.read_csv('./TAU-urban-acoustic-scenes-2020-mobile-development.meta/TAU-urban-acoustic-scenes-2020-mobile-development/meta.csv',sep='\t')
print(total)

meta = pd.DataFrame()
meta['mixture_audio_filename'] = total['filename'].str.split('/',n=1).str[1]
meta['bg_classname']=total['scene_label']
meta['bg_path']=total['filename']
meta['ebr']=0
meta['event_present']=False
meta['event']=''
meta['event_actual']=''
meta['event_length']=10.0

print(meta)

meta.to_csv('meta_scenes.csv',index=False)

