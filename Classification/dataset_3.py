import torch 
import csv 
import h5py
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import os
import numpy as np 
import cv2
from tqdm import tqdm 
import torch.nn.functional as F
#import albumentations as A
#from albumentations.pytorch.transforms import ToTensorV2
import time 
import random

class HDF5Dataset(Dataset):
    def __init__(self, data_path, patient_ids,num_patient=60,transform=None,style_model=None,mode='train', classification_type=False):
        self.data_path = data_path
        self.num_patient = num_patient
        self.patient_ids = [(patient.decode('utf-8'))+'_dic_msk' for patient in patient_ids]
        self.num_slices = {}
        self.data_cache = {}
        self.cache_size = 0
        self.transform = transform
        self.patient_slices = []
        self.mode = mode
        self.classification_type=classification_type

        with h5py.File(self.data_path, 'r') as f:
            self.patient_slices = []
            for patient_id in self.patient_ids:
                shape = f[patient_id][0].shape
                slice_indices = range(shape[0])
                self.patient_slices.extend([(patient_id, slice_index) for slice_index in slice_indices])
        
        
        self._load_patients_to_cache(self.patient_ids[0])

    def __getitem__(self, index):
        
        patient_id, slice_index = self.patient_slices[index]
        if patient_id not in self.data_cache : 
            self._flush_cache()
            self._load_patients_to_cache(patient_id)
        try :   
            image = self.data_cache[patient_id][0][slice_index]
            mask = self.data_cache[patient_id][1][slice_index]         
            image = self._preprocess_image(np.array(image))            
            mask = self._preprocess_mask(np.array(mask))
            
            return image,mask

        except KeyError : 
            print(f"Patient ID {patient_id} not found in data cache.")


    def __len__(self):
        return len(self.patient_slices)

    def _load_patients_to_cache(self,start_patient): 
        start_index = int(self.patient_ids.index(start_patient))
        end_index = len(self.patient_ids)  if start_index + self.num_patient >= len(self.patient_ids) else start_index + self.num_patient  
        
        with h5py.File(self.data_path, 'r') as f:
            for i in tqdm(range(start_index,end_index)):
                patient_id = self.patient_ids[i]
                image = f[patient_id][0][:]
                mask = f[patient_id][1][:]

                IndexShuffle = np.arange(mask.shape[0])
                np.random.shuffle(IndexShuffle)
                image = image[IndexShuffle]
                mask = mask[IndexShuffle]
                self.data_cache[patient_id] = (image,mask)
            print(self.data_cache.keys())

    def _flush_cache(self):       
        self.data_cache.clear()
        return True
    
    def _preprocess_mask(self,mask) :
        if (self.classification_type):
            if (255 in mask):
                return torch.tensor([1]) #classification en patho
            else:
                return torch.tensor([0]) #classification en sain
        else: #segmentation semantique
            mask = np.where(mask==100,1,mask) 
            mask = np.where(mask==200,2,mask)
            mask = torch.as_tensor(mask,dtype=torch.long)
            return mask
    
    def _preprocess_image(self,image) : 
        dt_aug= random.random()
        image = image.astype(np.uint8)
        image = torch.as_tensor(image/255.,dtype=torch.float32) 
        image= (image-0.5)/0.5 #ajout de Zscore normal
        return image.unsqueeze(0)     
    
    def _get_thresh_img(self,img) :
        _, thresh = cv2.threshold(img, 230, 255, cv2.THRESH_TOZERO)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel)
        thresh = cv2.dilate(thresh, kernel)
        return thresh
    
    def _preprocess_mask_v2(self,bone,mask) :
        bone = np.where(bone<230,0,255)
        mask = mask + bone
        mask = np.where(mask==100,1,mask) 
        mask = np.where(mask==200,2,mask)
        mask = np.where(mask==255,3,mask)

        # special condition if the addtion bone+mask is equal to 455 or 355 
        mask = np.where(mask==355,1,mask) 
        mask = np.where(mask==455,2,mask)
        return mask
     
    
    def _preprocess_mask_v3(self,mask) :
        _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        final_mask = self.generate_boundary(thresh)
        return final_mask
    


def SelectPatientsTrainVal(input_path, val_split):
    hf = h5py.File(input_path, "r")
    PatientsId = hf['patient_id'][0]
    print("patientId shape ", PatientsId.shape)	
    np.random.seed(42)
    np.random.shuffle(PatientsId)
    NPatients = PatientsId.shape[0]
    PatientsIdTrain = PatientsId[:int((1 - val_split) * NPatients + 0.5)]
    PatientsIdVal = PatientsId[int((1 - val_split) * NPatients + 0.5):]
    #PatientsIdTrain = PatientsId[:1]
    #PatientsIdVal = PatientsId[1:]

    hf.close()
    return np.array(PatientsIdTrain), np.array(PatientsIdVal)

def shuffle_batch(batch_images,batch_labels): 
    permutation = torch.randperm(batch_images.shape[0])

    # Use the permutation to shuffle the images and labels
    shuffled_images = batch_images[permutation]
    shuffled_labels = batch_labels[permutation] 
        
    return shuffled_images, shuffled_labels 

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




def save_image(batch): 
    n = batch[0].shape[0]
    images,masks = batch
    
    for i in range(n) :
        img = images[i]
        img = np.asarray(img)
        msk = masks[i]
        msk = np.asarray(msk)
        dt_aug= random.random()
        cv2.imwrite('images_out_test/image_out_'+str(i)+'.pgm',img.squeeze(0)*255)
        cv2.imwrite('images_out_test/mask_out'+str(i)+'.pgm',msk*100)
    
def main(): 
    path = './dataset/IRM_lung_dataset_256.hdf5'
    PatientsIdTrain,PatientsIdVal = SelectPatientsTrainVal(path, 0.2)
    
    
    
    db_train = HDF5Dataset(path,PatientsIdTrain[:5]) 
    #db_val = HDF5Dataset(path,PatientsIdVal,transform=transform) 
    
    loader_train = DataLoader(db_train, batch_size=32, shuffle=False,num_workers=5)
    #loader_val = DataLoader(db_val, batch_size=32, shuffle=False,num_workers=0)
    
    start_time = time.time() 
    total_size = 0
    for batch in loader_train: 
        images, true_masks = batch        
        save_image(batch)        
        
        print(torch.min(images),torch.max(images))
        print(torch.unique(true_masks))
        #assert torch.min(images) == 0 and torch.max(images)==1 
        print (torch.eq(torch.unique(true_masks),torch.tensor([0,1,2],dtype=torch.long)))
    
    end_time = time.time()
    
    print("time to load all data to gpu",str(round(end_time-start_time,1)) + "s")
    print(total_size)
    print(len(loader_train))
    



if __name__ == '__main__' : 
    main()