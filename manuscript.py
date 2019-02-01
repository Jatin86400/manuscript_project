import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os.path as osp
import json
import multiprocessing.dummy as multiprocessing
import random
import math
import cv2
from skimage.transform import resize
EPS = 1e-7 
def process_info(args):
    """
    Process a single json file
    """
    fname, opts = args
    
    with open(fname, 'r') as f:
        ann = json.load(f)
        f.close()
    examples = []
    skipped_instances = 0

    for instance in ann:
        components = instance['components']

        if 'class_filter'in opts.keys() and instance['label'] not in opts['class_filter']:
            continue
        
        candidates = [c for c in components if len(c['poly']) >= opts['min_poly_len']]

        if 'sub_th' in opts.keys():
            total_area = np.sum([c['area'] for c in candidates])
            candidates = [c for c in candidates if c['area'] > opts['sub_th']*total_area]

        candidates = [c for c in candidates if c['area'] >= opts['min_area']]

        if opts['skip_multicomponent'] and len(candidates) > 1:
            skipped_instances += 1
            continue

        instance['components'] = candidates
        if candidates:
            examples.append(instance)

    return examples, skipped_instances   

def collate_fn(batch_list):
    keys = batch_list[0].keys()
    collated = {}

    for key in keys:
        val = [item[key] for item in batch_list]

        t = type(batch_list[0][key])
        
        if t is np.ndarray:
            try:
                val = torch.from_numpy(np.stack(val, axis=0))
            except:
                # for items that are not the same shape
                # for eg: orig_poly
                val = [item[key] for item in batch_list]

        collated[key] = val

    return collated

class DataProvider(Dataset):
    """
    Class for the data provider
    """
    def __init__(self, opts, split='train', mode='train'):
        """
        split: 'train', 'train_val' or 'val'
        opts: options from the json file for the dataset
        """
        self.opts = opts
        self.mode = mode
        print('Dataset Options: ', opts)
        if self.mode !='tool':
            # in tool mode, we just use these functions
            self.data_dir = osp.join(opts['data_dir'], split)
            self.instances = []
            self.read_dataset()
            print('Read %d instances in %s split'%(len(self.instances), split))
    def read_dataset(self):
        data_list = glob.glob(osp.join(self.data_dir, '*/*.json'))
        data_list = [[d, self.opts] for d in data_list]
        print(len(data_list))
        pool = multiprocessing.Pool(self.opts['num_workers'])
        data = pool.map(process_info, data_list)
        pool.close()
        pool.join()
        print(len(data))

        print("Dropped %d multi-component instances"%(np.sum([s for _,s in data])))
        if(self.mode=='train'):
            self.instances = [instance for image,_ in data for instance in image]
        else:
            for image,_ in data:
                for instance in image:
                    if(instance['dec_per']>=0.5):
                        self.instances.append(instance)
               
        if 'debug' in self.opts.keys() and self.opts['debug']:
            self.instances = self.instances[:16]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.prepare_instance(idx)

    def prepare_instance(self, idx):
        """
        Prepare a single instance, can be both multicomponent
        or just a single component
        """
        instance = self.instances[idx]
        component = instance['components'][0]
       
        results = self.prepare_component(instance, component,instance['dec_per'])
        return results

    def prepare_component(self, instance, component,dec_per):
       
    #print "Prepare a single component within an instance"
        img = cv2.imread(instance['image_url'])
        label = instance['label']
        bbox = component['bbox']
        x0 = max(int(bbox[0]),0)
        y0 = max(int(bbox[1]),0)
        w = max(int(bbox[2]),0)
        h = max(int(bbox[3]),0)
        
        img = img[y0:y0+h,x0:x0+w]
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img  = resize(img,[64,832],anti_aliasing=True)
        img = torch.from_numpy(img)
        img = img.view(-1,1,64,832)
        img = img.float()
        poly = component["poly"]
     
        train_list = []
        pred_list = []
        train_list2 = []
        x0,y0,w,h = component["bbox"]
        if dec_per<=0.5:
            dec_len = math.ceil(dec_per*len(poly))
        else:
            dec_len = math.floor(dec_per*len(poly))
        #exponentially increasing eop values 
        start_eop = 0.1 #starting eop value
        steps = len(poly) #no of steps to devide
        gf = (int(1.0/start_eop))**(1.0/(steps-1)) #growth factor
        for step,i in enumerate(poly):
            list = []
            if step<len(poly)-dec_len:
                list.append(float(i[0]-x0)/float(w))
                list.append(float(i[1]-y0)/float(h))
        
                train_list.append(list)
                if step==len(poly)-dec_len-1:
                    list=[]
                    list.append(float(i[0]-x0)/float(w))
                    list.append(float(i[1]-y0)/float(h))
                    list.append(start_eop*gf**step)
                    train_list2.append(list)
            else:
                list.append(float(i[0]-x0)/float(w))
                list.append(float(i[1]-y0)/float(h))
                list.append(start_eop*gf**step)
                train_list2.append(list)
                pred_list.append(list) 
    #print pred_list
        pred_tensor = torch.FloatTensor(pred_list)
        train_tensor1 = torch.FloatTensor(train_list[:-1])# remove the last element, because it is part of train_tensor2
        train_tensor2 = torch.FloatTensor(train_list2[:-1])#remove the last element, because it is part of pred_tensor
        return_dict = {
                    "img" : img,
                    "input_tensor":train_tensor1,
                    "input_tensor2":train_tensor2,
                    "pred_tensor":pred_tensor,
                    "x0": x0,
                    "y0":y0,
                    "w" :w,
                    "h" : h,
                    "dec_per":dec_per,
                    "image_url":instance["image_url"],
                    }
        return return_dict
