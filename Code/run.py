from model_utils import  construct_traintest_data, construct_model_train_test
import torch, torchvision, os, argparse
import pandas as pd


parser = argparse.ArgumentParser(description="A script that takes an --id argument.")
parser.add_argument("--id", type=str, help="The ID parameter")
parser.add_argument("--cuda", type=int, help="The ID parameter")

args = parser.parse_args()



ID = args.id # --id 'cities_40_living_room'
cuda_id = args.cuda # --cuda_id 2
# torch.cuda.set_device(cuda_id)
os.chdir('/')
PATH = 'home/klimenko/nwl/new_living_visual/Methodology/CLASSIFICATION/'
DATA_DIR = PATH + 'datasets/'+ID+'.csv' # access training data

table = pd.read_csv(DATA_DIR)
label_str = list(table['class'].unique())


os.makedirs(PATH +'results/'+ID, exist_ok=True)
MODEL_PATH = PATH + 'results/'+ID+'/model_v1.pt' # save model path name
EPOCHS = 27
BATCH_SIZE = 24


train_iterator,valid_iterator,test_iterator, dataset = construct_traintest_data(DATA_DIR, BATCH_SIZE, label_str)
train_loss_list, val_loss_list = construct_model_train_test(train_iterator,valid_iterator,test_iterator,EPOCHS, MODEL_PATH, 'train', cuda_id, ID, label_str)
print('_____________________________')

