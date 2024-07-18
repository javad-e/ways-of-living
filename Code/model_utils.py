import os
import numpy as np
import torch, json
import torchvision
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from utils import set_parameter_requires_grad
from torch.utils.data import Dataset,DataLoader
import pandas as pd

from collections import Counter


from PIL import Image


PATH = 'home/klimenko/nwl/new_living_visual/Methodology/CLASSIFICATION/'
stats = ((0.5,0.5,0.5), (0.5,0.5,0.5))
pretrained_size = 450
SEED = 1 #1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class CustomCenterCrop:
    def __call__(self, img):
        crop_size = min(img.size)
        return torchvision.transforms.CenterCrop(crop_size)(img)
    

data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize(450),
                         torchvision.transforms.CenterCrop(450),                     
                         torchvision.transforms.Normalize(*stats,inplace=True)])




class ImageDataset(Dataset):
    def __init__(self, annotations_file, mode, label_str, transform=data_transforms):
        data = pd.read_csv(annotations_file)
#         subdata2 = data[data['mode']==mode].reset_index(drop=True)
        self.img_labels = data[data['mode']==mode].reset_index(drop=True)
        self.label_str = label_str
#         subdata3 = subdata2[subdata2['class']=='A'].reset_index(drop=True)
#         subdata4 = subdata2[subdata2['class']=='B'].reset_index(drop=True)

#         self.img_labels = pd.concat([subdata3,subdata4]).reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)
        


    def __getitem__(self, idx):
        

        img_path_1 = self.img_labels['path'][idx]
#         image1 = read_image('Datasets/reduction40/'+img_path_1).float()
        image1 = Image.open(img_path_1).convert('RGB')
#         try:
#             image1 = read_image(img_path_1).float()
#         except:
#             print('corrupted image')
#         print('edge') 
#         print(image1.size())

        
        image2 = self.transform(image1)
#         image22 = data_transforms_sub(image1)
#         image2 = torch.cat((image1, image2),0)
#         label_str=['A','B', 'C','D','S']
        LTR = self.img_labels['class'][idx]
        label_ = torch.tensor(self.label_str.index(LTR))    
        return image2, label_

#Image.open(r'Datasets/reduction40/'+img_path_1)#
def construct_traintest_data(DATA_DIR, BATCH_SIZE, label_str):
    num_classes = len(label_str)
    annotations_file = DATA_DIR
    
    
    #weighted sampler part
    data = pd.read_csv(annotations_file)
    img_labels_list = list(data[data['mode']=='train'].reset_index(drop=True)['class'])
    
    weights_list=[]
    n_count = Counter(img_labels_list)
    
    
    
    #obtain class counts
    for labelname in label_str:
        weightvalue = n_count[labelname]
        weightvalue2=weightvalue
        if weightvalue ==0:
            print('zero found')
            weightvalue2 = 1
        weights_list.append(weightvalue2)
        
    class_weights = [sum(weights_list)/weights_list[i] for i in range(len(weights_list))]
    
        
        
        
        
    
    
    
    train_dataset = ImageDataset(annotations_file, 'train', label_str)
    test_dataset = ImageDataset(annotations_file, 'test', label_str)
    val_dataset = ImageDataset(annotations_file, 'val', label_str)
    
    train_indices = list(range(len(train_dataset)))
    test_indices = list(range(len(test_dataset)))
    val_indices = list(range(len(val_dataset)))
    
    
    weights = [class_weights[label_str.index(img_labels_list[i])] for i in range(int(len(train_indices)))]
    
   
    
    #
    # class_counts = [9.0, 1.0]
    # num_samples = sum(class_counts)
    # labels = [0, 0,..., 0, 1] #corresponding labels of samples
    # class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
    # weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    # sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    #
    
    #print(weights)
    
    
    # Creating PT data samplers and loaders:
    train_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(len(train_indices)))# SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices) #SubsetRandomSampler(test_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,sampler=test_sampler)
    valid_iterator = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,sampler=valid_sampler)

    print(f'Number of training examples: {len(train_indices)}')
    print(f'Number of validation examples: {len(val_indices)}')
    print(f'Number of testing examples: {len(test_indices)}')
    
    return train_iterator,valid_iterator,test_iterator, train_dataset






def set_parameter_requires_grad(model, feature_extract):
	if feature_extract:
		for param in model.parameters():
			param.requires_grad = False







def define_model(model_choice, num_classes):
	if model_choice == 'resnet50':
		model = torchvision.models.resnet50(pretrained=True) # pick a model from the torchvisions collection
		set_parameter_requires_grad(model, feature_extract=False)
		IN_FEATURES = model.fc.in_features 
		OUTPUT_DIM = num_classes
		fc = torch.nn.Linear(IN_FEATURES, num_classes)
		model.fc = fc
        
        
	if model_choice == 'resnet18':
		model = torchvision.models.resnet18(pretrained=True) # pick a model from the torchvisions collection
		set_parameter_requires_grad(model, feature_extract=False)
		IN_FEATURES = model.fc.in_features 
		OUTPUT_DIM = num_classes
		fc = torch.nn.Linear(IN_FEATURES, num_classes)
		model.fc = fc

	elif model_choice == 'other':
            model = torchvision.models.densenet161(num_classes=num_classes)

	elif model_choice == 'ResNetSIAMSliced':
		model = ResNet_SIAM_Sliced(5)  
		set_parameter_requires_grad(model, feature_extract=False)
        
	elif model_choice == 'inception':
		model = torchvision.models.inception_v3(pretrained=True)
		set_parameter_requires_grad(model, feature_extract=False)

		num_ftrs = model.AuxLogits.fc.in_features
		model.AuxLogits.fc = torch.nn.Linear(num_ftrs, 2)

		num_ftrs = model.fc.in_features
		model.fc = torch.nn.Linear(num_ftrs,2)

            
	elif model_choice == 'densenet':
            
            model = torchvision.models.densenet161(pretrained=True)
            set_parameter_requires_grad(model, feature_extract=False)
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    

            
	return model




def compute_accuracy(y_pred, y):
    with torch.no_grad():
        batch_size = y.shape[0]
        y_pred = torch.argmax(y_pred, 1)
        corrects = torch.eq(y, y_pred).float().sum(0, keepdim = True)
        acc_val = corrects / batch_size
    return acc_val







def train(model, iterator, optimizer, criterion, scheduler, device):
	
	epoch_loss = 0
	epoch_acc = 0
	loss_list = []
	
	model.train()
	
	for (x, y) in iterator:
      
		x = x.to(device)
		y = y.to(device)
		optimizer.zero_grad()
		y_pred = model(x)
		loss = criterion(y_pred, y)
		loss_list.append(loss.item())
		acc_val =compute_accuracy(y_pred, y)
		print("loss",loss)
		loss.backward()
		optimizer.step()
		scheduler.step()
	
		
		epoch_loss += loss.item()

		epoch_acc += acc_val.item()
		
	epoch_loss /= len(iterator)
	epoch_acc /= len(iterator)
		
	return epoch_loss, epoch_acc, loss_list





def evaluate(model, iterator, criterion, device):
	
	epoch_loss = 0
	epoch_acc = 0
	loss_list = []
	
	model.eval()
	with torch.no_grad():
		
		for (x, y) in iterator:
			x = x.to(device)
			y = y.to(device)
			y_pred = model(x)
			loss = criterion(y_pred, y)
			acc_val =  compute_accuracy(y_pred, y)
			epoch_loss += loss.item()
			epoch_acc += acc_val.item()
			loss_list.append(loss.item())
		
	epoch_loss /= len(iterator)
	epoch_acc /= len(iterator)
		
	return epoch_loss, epoch_acc, loss_list




def get_predictions(model, iterator, device):
	embedding_layer = torch.nn.Sequential(*list(model.children())[:-1])
	model.eval()
	embedding_layer.eval()
	images = []
	embs = []   
	labels = []
	probs = []
	with torch.no_grad():

		for (x, y) in iterator:
			x = x.to(device)
			y_pred = model(x)
			emb = embedding_layer(x)            
			y_prob = torch.nn.functional.softmax(y_pred, dim = -1)
			top_pred = y_prob.argmax(1, keepdim = True)
			images.append(x.cpu())
			labels.append(y.cpu())
			embs.append(emb.cpu())
			probs.append(y_prob.cpu())

	images = torch.cat(images, dim = 0)
	labels = torch.cat(labels, dim = 0)
	probs = torch.cat(probs, dim = 0)
	emb_layer = torch.cat(embs, dim = 0)
	print(probs)

	return images, labels, probs, emb_layer






def construct_model_train_test(train_iterator,valid_iterator,test_iterator,EPOCHS, MODEL_PATH, train_test, cuda_id, ID, label_str):
    num_classes = len(label_str)
    torch.cuda.set_device(cuda_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    
    model = define_model('resnet50', num_classes) # create a model of our choice
    #device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu') #specify gpu usage
    #model = torch.nn.DataParallel(model).cuda()
    # model = model.to(device)
    model = model.to(device)
    START_LR = 1e-8#1e-7
    class_weights = [0.84, 0.93, 0.79, 0.98, 0.89, 0.84, 0.93, 0.8]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    #criterion = torch.nn.CrossEntropyLoss(weight = class_weights_tensor) # loss function
    #weight_tensor = torch.FloatTensor([1300/8*250,1300/8*250, 1300/8*100,1300/8*125, 1300/8*200, 1300/8*200, 1300/8*75, 1300/8*125])
    criterion = torch.nn.CrossEntropyLoss()#weight=class_weights_tensor
    criterion = criterion.to(device)

    
    
    if train_test == "train":
    
        params_to_update = []
        
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        else:
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    x=1


        optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001,
                                weight_decay=0.001) #was 0.001
        STEPS_PER_EPOCH = len(train_iterator)
        TOTAL_STEPS = (EPOCHS+1) * STEPS_PER_EPOCH
        MAX_LRS = [p['lr'] for p in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr = MAX_LRS,total_steps = TOTAL_STEPS)

        best_valid_loss = float('inf')
        best_valid_acc = float(0)
        train_loss_list = []
        valid_loss_list = []
        valid_acc_list = []
        for epoch in range(EPOCHS):

            train_loss, train_acc, train_loss_list_i = train(model, train_iterator, optimizer, criterion, scheduler, device)
            valid_loss, valid_acc, valid_loss_list_i = evaluate(model, valid_iterator, criterion, device)
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)
            with open(PATH +'results/'+ID+ '/train_loss.npy', 'wb') as f:
                np.save(f, train_loss_list)
        
            with open(PATH +'results/'+ID+ '/valid_loss.npy', 'wb') as f:
                np.save(f, valid_loss_list)
               
            
            with open(PATH +'results/'+ID+ '/valid_acc.npy', 'wb') as f:
                np.save(f, valid_acc_list)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), MODEL_PATH)

            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc*100:6.2f}% ')
            print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc*100:6.2f}% ' )
           
        with open(PATH +'results/'+ID+'/train_loss.npy', 'wb') as f:
            np.save(f, train_loss_list)
        
        with open(PATH  +'results/'+ID+ '/valid_loss.npy', 'wb') as f:
            np.save(f, valid_loss_list)
                
        return train_loss_list, valid_loss_list
        
        
    elif train_test == "test":
        
        
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH)), strict=False)
        test_loss, test_acc, loss_list = evaluate(model, test_iterator, criterion, device)

        images, labels, probs, emb_layer = get_predictions(model, test_iterator, device)
        pred_labels = torch.argmax(probs, 1)
        
        return images, labels, probs, pred_labels, test_loss, test_acc,model, emb_layer