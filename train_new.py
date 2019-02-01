import torch
import json
import os
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
from model import Model
import manuscript
from tqdm import tqdm
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("!!!!!!got cuda!!!!!!!")
else:
    print("!!!!!!!!!!!!no cuda!!!!!!!!!!!!")
def iou_from_poly(pred, gt, width, height):
    """
    Compute IoU from poly. The polygons should
    already be in the final output size

    pred: list of np arrays of predicted polygons
    gt: list of np arrays of gt polygons
    grid_size: grid_size that the polygons are in

    """
    width = int(width)
    height = int(height)
    masks = np.zeros((2, height, width), dtype=np.uint8)

    if not isinstance(pred, list):
        pred = [pred]
    if not isinstance(gt, list):
        gt = [gt]

    for p in pred:
        masks[0] = draw_poly(masks[0], p)

    for g in gt:
        masks[1] = draw_poly(masks[1], g)

    return iou_from_mask(masks[0], masks[1])
def accuracy_from_poly(pred, gt, width, height):
    """
    Compute IoU from poly. The polygons should
    already be in the final output size

    pred: list of np arrays of predicted polygons
    gt: list of np arrays of gt polygons
    grid_size: grid_size that the polygons are in

    """
    width = int(width)
    height = int(height)
    masks = np.zeros((2, height, width), dtype=np.uint8)

    if not isinstance(pred, list):
        pred = [pred]
    if not isinstance(gt, list):
        gt = [gt]

    for p in pred:
        masks[0] = draw_poly(masks[0], p)

    for g in gt:
        masks[1] = draw_poly(masks[1], g)

    return accuracy_from_mask(masks[0], masks[1])

def iou_from_mask(pred, gt):
    """
    Compute intersection over the union.
    Args:
        pred: Predicted mask
        gt: Ground truth mask
    """
    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)

    # true_negatives = np.count_nonzero(np.logical_and(np.logical_not(gt), np.logical_not(pred)))
    false_negatives = np.count_nonzero(np.logical_and(gt, np.logical_not(pred)))
    false_positives = np.count_nonzero(np.logical_and(np.logical_not(gt), pred))
    true_positives = np.count_nonzero(np.logical_and(gt, pred))

    union = float(true_positives + false_positives + false_negatives)
    intersection = float(true_positives)

    iou = intersection / union if union > 0. else 0.

    return iou
def accuracy_from_mask(pred, gt):
    """
    Compute intersection over the union.
    Args:
        pred: Predicted mask
        gt: Ground truth mask
    """
    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)

    # true_negatives = np.count_nonzero(np.logical_and(np.logical_not(gt), np.logical_not(pred)))
    true_negatives = np.count_nonzero(np.logical_and(np.logical_not(gt),np.logical_not(pred)))
    true_positives = np.count_nonzero(np.logical_and(gt, pred))
    accuracy = (true_positives)/(np.count_nonzero(gt))
    return accuracy
def draw_poly(mask, poly):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)

    cv2.fillPoly(mask, [poly], 255)

    return mask


def create_folder(path):
   # if os.path.exists(path):
       # resp = input 'Path %s exists. Continue? [y/n]'%path
    #    if resp == 'n' or resp == 'N':
     #       raise RuntimeError()
    
   # else:
     os.system('mkdir -p %s'%(path))
     print('Experiment folder created at: %s'%(path))
        
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    return args

def get_data_loaders(opts, DataProvider):
    print('Building dataloaders')

    dataset_train = DataProvider(split='train', opts=opts['train'],mode='train')
    dataset_val = DataProvider(split='train_val', opts=opts['train_val'],mode='train_val')

    train_loader = DataLoader(dataset_train, batch_size=opts['train']['batch_size'],
        shuffle = True, num_workers=opts['train']['num_workers'], collate_fn=manuscript.collate_fn)

    val_loader = DataLoader(dataset_val, batch_size=opts['train_val']['batch_size'],
        shuffle = False, num_workers=opts['train_val']['num_workers'], collate_fn=manuscript.collate_fn)
    
    return train_loader, val_loader

class Trainer(object):
    def __init__(self,args,opts):
        self.global_step = 0
        self.epoch = 0
        self.opts = opts
        create_folder(os.path.join(self.opts['exp_dir'], 'checkpoints_imgencoder3'))
       # Copy experiment file
        os.system('cp %s %s'%(args.exp, self.opts['exp_dir']))

        #self.writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train'))
        #self.val_writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train_val'))

        self.train_loader, self.val_loader = get_data_loaders(self.opts['dataset'], manuscript.DataProvider)
        self.model = Model(2,3,1,64,64,1,1,0,0)
        self.model = self.model.to(device)
        self.model.image_encoder.reload(self.opts['encoder_path'])
        # Allow individual options
        self.optimizer = optim.Adam(self.model.parameters(),lr = self.opts['lr'])
        for name, param in self.model.named_parameters():
            print(name)
        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opts['lr_decay'], 
            gamma=0.1)
        if args.resume is not None:
            self.resume(args.resume)
       
    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }

        save_name = os.path.join(self.opts['exp_dir'], 'checkpoints_imgencoder3', 'epoch%d_step%d.pth'\
        %(epoch, self.global_step))
        torch.save(save_state, save_name)
        print('Saved model')

    def resume(self, path):
        self.model.load_state_dict(torch.load(path)["state_dict"])
        save_state = torch.load(path, map_location=lambda storage, loc: storage)
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        self.lr_decay.load_state_dict(save_state['lr_decay'])

        print('Model reloaded to resume from Epoch %d, Global Step %d from model at %s'%(self.epoch, self.global_step, path)) 

    def loop(self):
        for epoch in range(self.epoch, self.opts['max_epochs']):
            self.epoch = epoch
            self.save_checkpoint(epoch)
#            self.lr_decay.step()
#            print 'LR is now: ', self.optimizer.param_groups[0]['lr']
            self.train(epoch)
    def change_data(self,data):
        input_batch = data["input_tensor"]
        input_batch2 = data["input_tensor2"]
        pred_batch = data["pred_tensor"]
        dec_per = data["dec_per"]
        x0s = data["x0"]
        y0s = data["y0"]
        ws = data["w"]
        hs = data["h"]
        tupel_list = []
        for i  in range(len(input_batch)):
            list = [input_batch[i],input_batch2[i],pred_batch[i],x0s[i],y0s[i],ws[i],hs[i],dec_per[i]]
            tupel_list.append(list)
        tupel_list = sorted(tupel_list,reverse=True,key=lambda x:len(x[0]))
        input_batch = []
        input_batch2 = []
        pred_batch =[]
        x0s = []
        y0s = []
        ws = []
        hs = []
        dec_per = []
    
        for x in tupel_list:
            input_batch.append(x[0])
            input_batch2.append(x[1])
            pred_batch.append(x[2])
            x0s.append(x[3])
            y0s.append(x[4])
            ws.append(x[5])
            hs.append(x[6])
            dec_per.append(x[7])
        return input_batch,input_batch2,pred_batch,x0s,y0s,ws,hs,dec_per
            
        
    def train(self, epoch):
        print('Starting training')
        self.model.train()
        losses = []
        ious = []
        accum = defaultdict(float)
        # To accumulate stats for printing
        for step, data in enumerate(self.train_loader):     
            if self.global_step % self.opts['val_freq'] == 0:
                self.validate()
        #self.model.train()
                self.save_checkpoint(epoch)
            img = data['img']
            img = torch.cat(img)
            img = img.view(-1,1,64,832)
            #the input to the encoder
            input_batch = data["input_tensor"]
            #the input to the decoder in training mode
            input_batch2 = data["input_tensor2"]
            #the predicted polygon 
            pred_batch = data["pred_tensor"]
            input_batch = torch.cat(input_batch)
            input_batch2 = torch.cat(input_batch2)
            pred_tensor = torch.cat(pred_batch)
            # Forward pass
            en_output,output = self.model(img.cuda(),input_batch.cuda(),input_batch2.cuda(),"train")
            output = output.cpu()
            en_output = en_output.cpu()
            #mse loss function
            loss_fn1 = nn.MSELoss()
            #mse loss for x,y part of output
            mse = 10*loss_fn1(output[:,:-1],pred_tensor[:,:-1]) 
            #mse loss for the eop value part of output, only the name is entropy, loss is mse
            entropy = loss_fn1(output[:,-1],pred_tensor[:,-1])
            #mse loss for the encoder output's x,y part
            en_mse = loss_fn1(en_output[:,:-1],input_batch2[0,:-1].view(-1,2))
            en_entropy = loss_fn1(en_output[:,-1],input_batch2[0,-1])
            #total loss is addition of all
            loss =  mse +entropy + en_mse + en_entropy
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            accum['loss'] += float(loss.item())
            accum['length'] += 1
            accum['mse'] += float(mse.item())
            accum['entropy']+=float(entropy.item())
            accum['en_mse'] += float(en_mse.item())
            accum['en_entropy']+=float(en_entropy.item())
            if(step%self.opts['print_freq']==0):
                # Mean of accumulated values
                for k in accum.keys():
                    if k == 'length':
                        continue
                    accum[k] /= accum['length']

                print("[%s] Epoch: %d, Step: %d, Loss: %f, mse: %f, entropy: %f, en_mse: %f, en_entropy: %f"%(str(datetime.now()), epoch, self.global_step, accum['loss'],accum['mse'],accum['entropy'],accum['en_mse'],accum['en_entropy']))
                accum = defaultdict(float)

            del(output)
            self.global_step += 1
        avg_epoch_loss = 0.0
        for i in range(len(losses)):
            avg_epoch_loss += losses[i]
        avg_epoch_loss = avg_epoch_loss/len(losses)
        print("Average Epoch %d loss is : %f"%(epoch,avg_epoch_loss))
    def validate(self):
        print('Validating')
        self.model.eval()
        ious = [0]*18
        accuracys = [0]*18
        count = [0]*18
        with torch.no_grad():
            for step, data in enumerate(tqdm(self.val_loader)):
                img = data["img"]
                input_batch = data["input_tensor"]
                input_batch2 = data["input_tensor2"]
                pred_batch = data["pred_tensor"]
                dec_per = data["dec_per"]
                ws = data["w"]
                hs = data["h"]
                img = torch.cat(img)
                img = img.view(-1,1,64,832)
                #input for the encoder
                input_batch = torch.cat(input_batch)
                #input for the decoder in training mode
                input_batch2= torch.cat(input_batch2)
                
                if(len(input_batch)==0):
                    continue
                en_output,output= self.model(img.cuda(),input_batch.cuda(),input_batch2.cuda(),"test")
                output = output.cpu()
                output = output[:,:-1]
                pred_batch = torch.cat(pred_batch)
                pred_batch = pred_batch[:,:-1]
                en_output = en_output.cpu()
                en_output = en_output[:,:-1]
                #generate polygon from the ipnuts and output
                list = []
                list.append(input_batch)
                list.append(en_output.view(1,2))
                list.append(output)
                output = torch.cat(list)
                #gnenerate polygon from the inputs and ground truths
                list = []
                list.append(input_batch)
                list.append(input_batch2[0,:-1].view(1,2))
                list.append(pred_batch)
                pred = torch.cat(list)
                output = output.numpy()
                #remove the eop values
                
              
                pred = pred.numpy()
                   
                #convert vertices in coordinates with respect to x0,y0 as origin
                for i in output:
                    i[0] = i[0]*ws[0]
                    i[1] = i[1]*hs[0]
                    if i[0]<=0:
                        i[0] = 0
                    if i[1]<=0:
                        i[1] = 0
                    if i[0]>ws[0]:
                        i[0] = ws[0]
                    if i[1]>hs[0]:
                        i[1] = hs[0]     
                for i in pred:
                    i[0] = i[0]*ws[0]
                    i[1] = i[1]*hs[0]
                
                output = output.astype(int)
                pred = pred.astype(int)
                #calculate iou of the predicted and groundtruth
                iou = iou_from_poly(output,pred,ws[0],hs[0])
                accuracy = accuracy_from_poly(output,pred,ws[0],hs[0])
                ious[int((dec_per[0]*100)/5-1)]+=iou
                accuracys[int((dec_per[0]*100)/5-1)]+=accuracy
                count[int((dec_per[0]*100)/5-1)]+=1
                #delete output to save space
                del(output)
            avg_loss = [0]*18
            avg_accuracy = [0]*18
            for i in range(len(ious)):
                if(count[i]!=0):
                    avg_loss[i] = float(ious[i]/count[i])
                    avg_accuracy[i] = float(accuracys[i]/count[i])
                    print("avg iou for %f dec_per is %f and accuracy is %f"%((i+1)*0.05,avg_loss[i],avg_accuracy[i]))
        self.model.train()
if __name__ == '__main__':
    args = get_args()
    opts = json.load(open(args.exp, 'r'))
    opts['max_epochs'] = 50
    print(opts)
    trainer = Trainer(args,opts)
    trainer.loop()
