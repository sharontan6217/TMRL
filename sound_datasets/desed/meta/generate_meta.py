import pandas as pd

labels = pd.read_csv('../Ground-truth/Ground-truth/ground_truth_eval.tsv',delimiter='\t')
print (labels)

mapping = pd.read_csv('../Ground-truth/Ground-truth/mapping_file_eval.tsv',delimiter='\t')
print(mapping)

duration = pd.read_csv('../Ground-truth/Ground-truth/eval_durations.tsv',delimiter='\t')
print(duration)

total = labels.merge(mapping, how='inner', on = 'filename').merge(duration,how='inner',on='filename')

print(total)

meta = pd.DataFrame()
meta['mixture_audio_filename'] = total['public_file']
meta['wav_name'] = total['public_file']
meta['bg_classname']=''
meta['bg_path']=''
meta['ebr']=0
meta['event_present']=True
meta['event']=total['event_label']
meta['event_actual']=total['event_label']
meta['event_length']=total['duration']

print(meta)

meta.to_csv('meta.csv',index=False)

