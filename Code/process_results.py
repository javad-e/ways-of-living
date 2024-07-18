import torchvision.transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch, os
import torch.nn as nn
import numpy as np
import torch, json
import torchvision
from utils import set_parameter_requires_grad, denormalize, show_batch, display_img
from model_utils import define_model, compute_accuracy, train, evaluate, get_predictions, construct_traintest_data, construct_model_train_test,set_parameter_requires_grad 
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import models, transforms
from collections import Counter
import shutil
torch.cuda.empty_cache()
import os
####
import sklearn.decomposition # PCA
import sklearn.manifold # MDS, t-SNE
import time
#####
import PIL
from PIL import Image
count = 0
from datetime import datetime
stats = ((0.5,0.5,0.5), (0.5,0.5,0.5))
import re
import re
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.CenterCrop(300),
                         torchvision.transforms.Resize(300),                     
                         torchvision.transforms.Normalize(*stats,inplace=True)])

from torchvision.io import read_image
def remove_non_numeric(string):
    pattern = r"[^0-9.]"
    return re.sub(pattern, "", string)


device = torch.device('cuda:1') #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.chdir('/')


DATASET_ID = 'small3'

def process_results_to_csv(ROOM, MODE):
    
    DATA_DIR = PATH + 'datasets/cities_'+DATASET_ID+'_'+ROOM+'.csv'
    MODEL_PATH = PATH + 'results/cities_'+DATASET_ID+'_'+ROOM+'/model_v1.pt'
    BATCH_SIZE = 4
    
    table = pd.read_csv(DATA_DIR)
    label_str = list(table['class'].unique())



    class_names = label_str
    train_iterator,valid_iterator,test_iterator, dataset = construct_traintest_data(DATA_DIR, BATCH_SIZE, label_str)
    torch.cuda.empty_cache()
    model = torchvision.models.resnet50(pretrained=True)
    set_parameter_requires_grad(model, feature_extract=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(label_str))
    
    with torch.no_grad():
        model.load_state_dict(torch.load(MODEL_PATH))
    
    latent_model = nn.Sequential(*list(model.children())[:-1])
    latent_model.eval()
    model.to(device)
    latent_model.to(device)
    print('_____________________________')


    vectorlist = []
    vectorlist_rgb = []
    labellist = []
    income_list = []
    min_nights_list=[]
    clutter_seg = []
    clutter_rgb = []
    counter = 0

    now1 = datetime.now()
    dataframe2 = pd.read_csv(DATA_DIR)
    dataframe2 = dataframe2[dataframe2['mode']==MODE].reset_index(drop=True)
    dataframe2['latent'] =np.nan
    dataframe2['pred_city'] =np.nan
    dataframe2['pred_label'] =np.nan
    dataframe2['labels'] =np.nan
    
    X = 0
    X_range = list(range(0, len(dataframe2)+1, 200))
    for X in X_range:
        print('NEW XRANGE_____________')
        dataframe22 = dataframe2[X:X+200]
        dataframe22.reset_index(drop=True, inplace=True)
        
        
        for i in range(len(dataframe22)):#len(dataframe2)
            
            print(MODE+'  '+ROOM)
            print('II INDEX '+str(i))
            if dataframe22['class'][i] in class_names:
        #         try:
                basepath = dataframe22['path'][i]
                image1 = Image.open(basepath).convert('RGB')
                #image1 = read_image(basepath).float()
                image2 = transform(image1)
                image3 = image2.unsqueeze(0)
                image3 = image3.to(device)
                vec = np.asarray(latent_model(image3).cpu().detach())[0,:,0,0]
                list_string = ', '.join(map(str, vec))
                dataframe22.loc[i, 'latent'] = list_string
                print(i)
                pred = model(image3)
                y_prob = torch.nn.functional.softmax(pred, dim = -1)
                top_pred = y_prob.argmax(1, keepdim = True)
                citypred = label_str[int(top_pred.item())]
                dataframe22.loc[i, 'pred_city'] = citypred
          
                dataframe22.loc[i, 'pred_label'] = int(top_pred.item())
        #         except:
        #             print('not found')


        now2 = datetime.now()-now1
        print('TIME FOR ALL IMG: ',now2)
        
        DATAFRAME_PATH = PATH + 'results/cities_'+DATASET_ID+'_'+ROOM+'/processed_model_v1_'+MODE+'.csv'
        if X == 0:
            dataframe22.to_csv(DATAFRAME_PATH)
        else:
            dataframe22_base =pd.read_csv(DATAFRAME_PATH)
            combined_df = dataframe22_base.append(dataframe22, ignore_index=True)
            combined_df.to_csv(DATAFRAME_PATH)
            
        print(count)
    
    
    
    PATH2 = PATH + 'results/cities_'+DATASET_ID+'_'+ROOM+'/processed_model_v1_'+MODE+'.csv'
    print('PROCESSING TSNE_______________')
    dataframe = pd.read_csv(PATH2)
    label_str=os.listdir('/media/data_16T/nwl/processed_data/classified_rooms/airbnb/complete')
    label_str = [string[:-4] for string in label_str]
    class_names = label_str
    dataframe = dataframe.dropna(subset=['latent'])
    dataframe = dataframe[dataframe['class'].isin(class_names)]


    vectorlist_str = list(dataframe['latent'])
    vectorlist = list(map(lambda x: list(map(float, x.split(', '))), vectorlist_str))
    labellist = list(dataframe['class'])
    pathlist = dataframe['path']

    tree = np.float64(np.asarray(vectorlist))
    cmap = plt.get_cmap('Reds')
    start = time.time()
    tsne_operator = sklearn.manifold.TSNE(n_components=2)
    tree_tsne = tsne_operator.fit_transform(tree)
    end = time.time()
    print("Embedded t-SNE in {:.2f} seconds.".format(end-start))

    data = {
        'path': pathlist,
        'label': labellist,
        'x': tree_tsne[:,0],
        'y': tree_tsne[:,1],
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    df.to_csv(PATH + 'results/cities_'+DATASET_ID+'_'+ROOM+'/processed_model_v1_'+MODE+'_TSNE.csv')



PATH = '/home/klimenko/nwl/new_living_visual/Methodology/CLASSIFICATION/'

# RUN THE ANALYSES
#process_results_to_csv('bathroom', 'test')
#process_results_to_csv('kitchen', 'test')
#process_results_to_csv('living_room', 'test')
process_results_to_csv('kitchen', 'test')

# process_results_to_csv('bathroom', 'val')
# process_results_to_csv('kitchen', 'val')
# process_results_to_csv('living_room', 'val')

# process_results_to_csv('bathroom', 'train')
# process_results_to_csv('kitchen', 'train')
# process_results_to_csv('living_room', 'train')
