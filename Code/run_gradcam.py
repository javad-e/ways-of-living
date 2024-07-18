import torch, os, re, shutil, torchvision, json, argparse,torchvision.transforms
from torchvision.io import read_image
import torch.nn as nn
import numpy as np
from skimage.transform import resize
from utils import set_parameter_requires_grad, denormalize, show_batch, display_img
from model_utils import define_model, construct_traintest_data, construct_model_train_test,set_parameter_requires_grad 
import pandas as pd
from mit_semseg.semseg import segment_image, GradCAM, visualize_result


# import arguments
###############################################################
parser = argparse.ArgumentParser(description="A script that takes an --id argument.")
parser.add_argument("--id", type=str, help="The ID parameter")
parser.add_argument("--cuda", type=int, default=0 ,help="The cuda device parameter")
parser.add_argument("--start", type=int, default=0 ,help="The cuda device parameter")
args = parser.parse_args()

ID = args.id # --id 'cities_40_living_room'
cuda_id = args.cuda
a_start = args.start
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(cuda_id)


mode_list = ['test']
savename = 'gradcam_results_threshold005'   
###############################################################







# import needed files
###############################################################
os.chdir('/')
BATCH_SIZE=4
PATH = '/home/klimenko/nwl/new_living_visual/Methodology/CLASSIFICATION/'
#ID = 'small2_living_room'
DATA_DIR = PATH + 'datasets/cities_'+ID+'.csv'
MODEL_PATH = PATH + 'results/cities_'+ID+'/model_v1.pt'
analysis_file = pd.read_csv('/home/klimenko/nwl/new_living_visual/Methodology/CLASSIFICATION/results/cities_'+ID+'/processed_model_v1_test_TSNE.csv')
table = pd.read_csv(DATA_DIR)
label_str = list(table['class'].unique())
train_iterator,valid_iterator,test_iterator, dataset = construct_traintest_data(DATA_DIR, BATCH_SIZE, label_str)
###############################################################


# special function to resize objects even if they consist of label patches from segmentation
###############################################################
def resize_and_center_crop(array):
    width, height = array.shape[:2]
    scaling_factor = 450 / min(width, height)
    new_width = round(width * scaling_factor)
    new_height = round(height * scaling_factor)
    resized_array = resize(array, (new_width, new_height), order=0, anti_aliasing=False, preserve_range=True)  
    y,x = resized_array.shape
    startx = x//2 - 450//2
    starty = y//2 - 450//2    
    return resized_array[starty:starty+450, startx:startx+450]
###############################################################






# arrange cities in the order they appear in the dataset
###############################################################
def list_to_ordered_set(input_list):
    ordered_set = {}
    for item in input_list:
        ordered_set[item] = None  # Using the dictionary keys to maintain order
    return list(ordered_set.keys())
###############################################################


# define model and load weights from file, set up according GradCAM model
###############################################################
model = torchvision.models.resnet50(pretrained=True)
set_parameter_requires_grad(model, feature_extract=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(label_str))
model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
model.to(device)
grad_cam = GradCAM(model, 'layer4')
###############################################################



# for each city in the dataset, compute centroids and see how far each point is from a centroid of a particular city
###############################################################
citylist = list_to_ordered_set(list(analysis_file['label']))
analysis_df = pd.DataFrame()
total_mass = len(analysis_file)
com_x = np.sum(analysis_file['x']) / total_mass
com_y = np.sum(analysis_file['y']) / total_mass

analysis_file['distance_to_com']=0.0
for city in citylist:
    analysis_sub = analysis_file[analysis_file['label']==city]
    analysis_sub['distance_to_com'] = np.sqrt((analysis_sub['x'] - com_x)**2 + (analysis_sub['y'] - com_y)**2)
    analysis_sub['percentile'] = analysis_sub['distance_to_com'].rank(pct=True)
    analysis_df = analysis_df.append(analysis_sub, ignore_index=True)
###############################################################    
    
  
# load a dataset so that we can add additional columns with our gradcam detections per object
###############################################################

label_df = pd.read_csv('/home/klimenko/nwl/new_living_visual/Methodology/classifier_grad_cam/data/object150_info.csv')
object_names = list(label_df['Name'])
object_names.remove('path')
for column_name in object_names:
    analysis_df[column_name] = 0
    
###############################################################





    
    
    
    
# define transform parameters    
###############################################################    
stats = ((0.5,0.5,0.5), (0.5,0.5,0.5))
transform = torchvision.transforms.Compose([torchvision.transforms.Resize(450),torchvision.transforms.CenterCrop(450),
                                              torchvision.transforms.Normalize(*stats,inplace=True)])
###############################################################




# loop over dataset and run gradcam/detections. We want to process dataframe in chunks L=100 not to overload CPU
############################################################### 
OUTPUT_PATH = PATH + 'results/cities_'+ID+'/'+savename+'.csv'


X = 0
interval = 200
interval_A = 8000
X_range = list(range(a_start, len(analysis_df)+1, interval))
#A_range =  list(range(a_start, len(dataframe2)+1, interval_A)) obsolete



for X in X_range:
    # make sure OUTPUT_PATH changes every 8000 points so that we do not overload memory
#     if X in A_range:
#         OUTPUT_PATH = PATH + 'results/cities_'+ID+'/'+savename+'_'+str(X)+'.csv'
        
    print(X)
    print('NEW XRANGE_____________')
    dataframe2_sub = analysis_df[X:X+interval]
    dataframe2_sub.reset_index(drop=True, inplace=True)
    
    for index in range(len(object_names)):
        dataframe2_sub[object_names[index]]=0
        

    for i in range(len(dataframe2_sub)):
        print(i+X)
        


        basepath = list(dataframe2_sub['path'])[i]
        print(basepath)
        segment_mask = resize_and_center_crop(segment_image(basepath))
        
        image1 = read_image(basepath).float()
        image2 = transform(image1)
        image3 = image2.unsqueeze(0).to(device)
        heatmap = grad_cam(image3)
        heatmap_mask = heatmap > 0.05

        for index in range(len(object_names)):#
            object_name = object_names[index]
            condmask = np.where(segment_mask == index, 1, 0)
            score_uniform = np.sum(condmask*heatmap_mask)/((np.sum(condmask)+1))
            dataframe2_sub.loc[i, object_name] = score_uniform     
            ############################################################### 
            


    if X == 0:
        dataframe2_sub.to_csv(OUTPUT_PATH, index=False)
        
        
    else:
        dataframe2_base =pd.read_csv(OUTPUT_PATH,index_col=0) 
        
        combined_df = dataframe2_base.append(dataframe2_sub).reset_index(drop=True)
        combined_df.to_csv(OUTPUT_PATH)
        
        del dataframe2_sub
        del combined_df
        

        
############################################################### 
