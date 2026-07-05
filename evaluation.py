
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
import csv
import glob
import scipy
from scipy import fft
from scipy.io import wavfile
from scipy.stats import spearmanr,pearsonr
from libauc.metrics import pauc_roc_score
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, hamming_loss,jaccard_score,roc_auc_score,roc_curve, precision_recall_curve,auc, f1_score,silhouette_score,normalized_mutual_info_score, adjusted_rand_score
from IPython.display import display, Audio
import plotnine
from plotnine import *
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import utils
from utils import utils




def evaluationMatrix(df_representation,classification_type,event_list,random_int,opt):
    log_dir = opt.log_dir
    df_representation['target_event']=''

    files = set(df_representation['wav_name'])
    #print(files)
    
    df_event=pd.DataFrame()
    if (classification_type=='event'or classification_type=='both')==True:
        df_representation=df_representation.drop('index',axis=1)
        for file in files:
            #print(file)
            df_event_predict = df_representation[df_representation['wav_name']==file].reset_index()
            '''
            for target_event in event_list:
                if target_event in file:
                    df_event_predict['target_event']=target_event
            '''
            predicted_events = set( df_event_predict['event_predict'])
            df_event_predict.to_csv('predict_'+file+'.csv')
            #print(file,predicted_events)
            df_event_predict['event_present_predict']=False
            for i in range(len(df_event_predict)):
                df_event_predict['event_actual'][i]=df_event_predict['event_actual'][i].replace("_"," ")
                #print(df_event_predict['event_present'][i],df_event_predict['target_event'][i],df_event_predict['event_predict'].values)
                if df_event_predict['event_present'][i]==True and df_event_predict['event_actual'][i].lower() in predicted_events:
                    df_event_predict['event_present_predict'][i]=True

            df_event=pd.concat((df_event,df_event_predict),axis=0).drop('index',axis=1)
        df_event.to_csv('result.csv')
        df_event_result = df_event[['wav_name','event_present','event_present_predict','event_actual']]
        df_event_result = df_event_result.drop_duplicates()

        df_event_result = df_event_result.reset_index()
        wav_name_array = df_event_result['wav_name']
        event_array  = df_event_result['event_actual']
        #print(df_event_result)
        actual = [] 
        predict = []
        for i in range(len(df_event_result)):
            #print(i,df_event_result['event_present'][i],df_event_result['event_present_predict'][i])
            if df_event_result['event_present'][i]==True:
                actual.append(1)
            else:
                actual.append(0)
            if df_event_result['event_present_predict'][i]==True:
                predict.append(1)
            else:
                predict.append(0)
        actual=np.array(actual)
        predict=np.array(predict)
    elif (classification_type=='environment'or classification_type=='both')==True:
        df_environment = df_representation[['wav_name','bg_classname','env_selected','score_selected','environment_top5','env_level1']]
        print(df_environment,df_representation)
        #df_environment = df_environment.drop_duplicates('wav_name',inplace=True)
        df_environment =  df_environment[~df_environment['wav_name'].duplicated()].reset_index()
        print(df_environment)
        wav_name_array = df_environment['wav_name']
        env_array = df_environment['bg_classname']
        accuracy_score_selected=0.0000
        accuracy_score_top5=0.0000
        accuracy_score_level1=0.0000
        total_score = len(df_environment)
        actual = df_environment['bg_classname'].values
        predict =  df_environment['env_level1'].values
        for i in range(len(df_environment)):
            actual_environment=df_environment['bg_classname'][i]
            print(actual_environment)

            if actual_environment in df_environment['env_selected'][i]:
                accuracy_score_selected+=1
            if actual_environment in df_environment['environment_top5'][i]:
                accuracy_score_top5+=1
            if actual_environment in df_environment['env_level1'][i]:
                accuracy_score_level1+=1
        accuracy_score_selected = accuracy_score_selected/total_score
        accuracy_score_top5 = accuracy_score_top5/total_score
        accuracy_score_level1 = accuracy_score_level1/total_score


    matrix=pd.DataFrame()
    matrix['wav_names']=[wav_name_array]
    matrix['random_int']=random_int
    matrix['classification_type']=classification_type
    
    if (classification_type=='event'or classification_type=='both')==True:
        eventScore=math.sqrt(mean_squared_error(actual,predict)) 
        print('Train Score: %.5f RMSE' % (eventScore))




        actual_2d = actual.reshape(-1,1)
        predict_2d = predict.reshape(-1,1)
        print(actual_2d)
        print(predict_2d)
        auc_value=roc_auc_score(actual_2d,predict_2d,average='macro')
        pauc_value=pauc_roc_score(actual,predict,max_fpr=0.3,min_tpr=0.7)

        cm=confusion_matrix(actual,predict)
        ari=adjusted_rand_score(actual,predict)
        nmi=normalized_mutual_info_score(actual,predict)
        #si=silhouette_score(actual_2d,predict_2d,metric='euclidean')
        fmeasure=f1_score(actual,predict,average='micro')
        ac_score=accuracy_score(actual,predict)
        #ci = utils.ci(actual,ac_score,1.96)

        print('auc: ',auc_value)
        print('pauc: ',pauc_value)
        print('cm_predict: ',cm)
        print('ARI: ',ari)
        print('NMI: ',nmi)
        #print('SI: ',si)
        print('F Measure: ',fmeasure)
        print('Accuracy: ', ac_score)
        mse= mean_squared_error(actual,predict,multioutput='raw_values')
        print('mse = ',mse)
        #print('ci = ',ci)   
        f= open(log_dir+'mpnet_'+str(random_int)+'_'+classification_type+'.txt','a') 
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
        matrix['ac_score']=ac_score
        matrix['f_measure']=fmeasure
        matrix['mse']=mse
        matrix['auc']=auc_value
        matrix['pauc']=pauc_value
        matrix['ari']=ari
        matrix['nmi']=nmi
        matrix['confusion_matrix']=[cm]
        matrix['event']=[event_array]

    elif (classification_type=='environment' or classification_type=='both')==True:
        matrix['accuracy_score_selected']=accuracy_score_selected
        matrix['accuracy_score_top5']=accuracy_score_top5
        matrix['accuracy_score_level1']=accuracy_score_level1
        matrix['environments']=[env_array]

    return matrix, actual,predict,wav_name_array
    

def visualize(matrix_total,actual_event_total,predict_event_total,random_int,event,classification_type,opt):
    graph_dir = opt.graph_dir

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
    plt.savefig(graph_dir+png_name_roc)
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
    plt.savefig(graph_dir+png_name_roc)
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
    fig.savefig(graph_dir+'graph/no_noise/'+png_name_line)
    plt.close()

    return fig