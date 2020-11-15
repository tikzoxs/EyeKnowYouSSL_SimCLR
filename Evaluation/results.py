import pandas as pd 
import csv
import glob

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix

for file_name in glob.glob('./'+'*.csv'):
    df = pd.read_csv(file_name, low_memory=False)
    truth=df['truth'].tolist()
    predicted=df['pred'].tolist()
    
    print(file_name)
    
    acc=accuracy_score(truth, predicted) 
    precision, recall, fscore, support = score(truth, predicted)
    
    print('accuracy: {}'.format(acc))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))    
    
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    
