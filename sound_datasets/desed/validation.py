import pandas as pd

gr = pd.read_csv('C:/Users/sharo/Documents/Github/TMRL/sound_datasets/data/Ground-truth/Ground-truth/ground_truth_eval.tsv',sep='\t')
print(gr)
test = pd.read_csv('C:/Users/sharo/Documents/Github/TMRL/sound_datasets/data/Ground-truth/Ground-truth/validation_groudtruth.csv')
mapping = pd.read_csv('C:/Users/sharo/Documents/Github/TMRL/sound_datasets/data/Ground-truth/Ground-truth/mapping_file_eval.tsv',sep='\t')
validation = test.merge(gr,how='left',on='filename').merge(mapping,how='left',on='filename')
duration = pd.read_csv('C:/Users/sharo/Documents/Github/TMRL/sound_datasets/data/Ground-truth/Ground-truth/eval_durations.tsv',sep='\t')
meta = duration.merge(gr,how='left',on='filename').merge(mapping,how='left',on='filename')
print(validation)
print(meta)
validation['wav_name']=validation['public_file']
validation['onset']=validation['onset_y']
validation['offset']=validation['offset_y']
validation['event_label']=validation['event_label_y']
vgr = validation[['wav_name','onset','offset','event_label']]
meta['wav_name']=meta['public_file']
meta = meta[['wav_name','duration']]
vgr.to_csv('validation_groundtruth.tsv',sep='\t',index=False)
meta.to_csv('meta.tsv',sep='\t',index=False)