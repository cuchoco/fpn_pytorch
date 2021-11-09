import torch
from torch.utils.data import Dataset

import numpy as np
import SimpleITK as sitk
import os 

class BrainDataset(Dataset):
    
    def __init__(self, ann, transform=None):
        self.ann = ann
        self.brain_classes = ('hemorrhage', 'fracture')
        self.transform = transform
        
        
    def reshape_image(self, img):
        img = np.squeeze(img)
        img = np.expand_dims(img, axis=2)

        return img

    def windowing(self, input_img, mode):

        if mode == 'hemorrhage':
            windowing_level = 40
            windowing_width = 160

        elif mode == 'fracture': 
            windowing_level = 600
            windowing_width = 2000

        elif mode == 'normal':
            windowing_level = 30
            windowing_width = 95

        density_low = windowing_level - windowing_width/2 # intensity = density
        density_high = density_low + windowing_width

        output_img = (input_img-density_low) / (density_high-density_low)
        output_img[output_img < 0.] = 0.           # windowing range
        output_img[output_img > 1.] = 1.

        return np.array(output_img, dtype='float32')
    
    def load_image(self, img_path):
        img = self.reshape_image(sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype('float32'))    
        img = np.concatenate([self.windowing(img, 'hemorrhage'), self.windowing(img, 'fracture'), self.windowing(img, 'normal')], axis=2)
        return img
    
    def __getitem__(self, index):
        root_path = self.ann['root_dir']
        dir_path = self.ann['images'][index]['dir']
        file_name = self.ann['images'][index]['file_name']
        
        img_path = os.path.join(root_path, dir_path, file_name)
        
        img = self.load_image(img_path)
        annot = self.load_annotations(index)
        
        sample = {'img': img, 
                  'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
    def __len__(self):
        return len(self.ann['images'])

    def num_classes(self):
        return len(self.brain_classes)

    def label_to_name(self, label):
        return self.brain_classes[label]
    
    
    def load_annotations(self, index):
        # get ground truth annotations
        
        cts = [i['category'] for i in self.ann['annotations'][index]['bbox_info']]
        bxs = [i['bbox'] for i in self.ann['annotations'][index]['bbox_info']]
        
        annotations = np.zeros((0, 5))
        
        # some images appear to miss annotations (like image with id 257034)
        if len(bxs) == 0:
            return annotations
            
        else:
            bxs= np.asarray(bxs)
            cts = np.asarray(cts)
            cts = np.expand_dims(cts , 1)
            annotations = np.concatenate((bxs, cts), axis=1)
            
        return annotations      
    
    
def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = torch.FloatTensor(annot)
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
    

    imgs = torch.from_numpy(np.stack(imgs, axis=0)).to(torch.float32)    
    imgs = imgs.permute(0, 3, 1, 2)

    return imgs, annot_padded