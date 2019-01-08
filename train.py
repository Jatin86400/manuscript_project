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
import torch.nn.utils.rnn as rnn
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
        create_folder(os.path.join(self.opts['exp_dir'], 'checkpoints'+str(opts['dec_per'])))

       # Copy experiment file
        os.system('cp %s %s'%(args.exp, self.opts['exp_dir']))

        #self.writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train'))
        #self.val_writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train_val'))

        self.train_loader, self.val_loader = get_data_loaders(self.opts['dataset'], manuscript.DataProvider)
        self.model = Model(4,1,64,64,1,1,0,0)
        self.model = self.model.to(device)
        # Allow individual options
        self.optimizer = optim.Adam(self.model.parameters(),lr = self.opts['lr'])
        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opts['lr_decay'], 
            gamma=0.1)
        self.inputdata = []
        self.preddata = []
        self.valdata = []
        self.predvaldata =[]
        self.inputdata2 = []
        self.valdata2 =[]
        self.dec_per_val=[]
        self.key=0
        self.valx = []
        self.valh=[]
        self.valw = []
        self.valy = []
    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }

        save_name = os.path.join(self.opts['exp_dir'], 'checkpoints'+str(self.opts['dec_per']), 'epoch%d_step%d.pth'\
        %(epoch, self.global_step))
        torch.save(save_state, save_name)
        print('Saved model')

    def resume(self, path):
        self.model.reload(path)
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
        if epoch==0:
            for step, data in enumerate(self.train_loader):     
                input_batch,input_batch2,pred_batch,x0s,y0s,ws,hs,dec_per = self.change_data(data)
                input_pack = rnn.pack_sequence(input_batch)
                self.inputdata.append(input_pack)
                input_batch2 = torch.cat(input_batch2)
                input_batch2 = input_batch2.to(device)
                pred_tensor = torch.cat(pred_batch).to(device)
                self.preddata.append(pred_tensor)
                self.inputdata2.append(input_batch2)
        
            # Forward pass
                output = self.model(input_pack.cuda(),input_batch2,"train")
                loss_fn1 = nn.MSELoss()
                loss_fn2 = nn.CrossEntropyLoss()
        
                mse = loss_fn1(output[:,:-2],pred_tensor[:,:-2]) 
    
                output1 = output[:,-1].view(-1,1)
                pred = pred_tensor[:,-1]
        
        #entropy = loss_fn2(torch.cat((1-output1,output1),dim=1),pred.type(torch.cuda.LongTensor))
                entropy = loss_fn1(output[:,-1],pred)
        
            #print 'mse %f'%mse
        #print 'entropy %f'%entropy
                loss =  10*mse + entropy
            # Backward pass
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                accum['loss'] += float(loss.item())
                accum['length'] += 1
                accum['mse'] += float(mse.item())
                accum['entropy']+=float(entropy.item())
                if(step%self.opts['print_freq']==0):
                # Mean of accumulated values
                    for k in accum.keys():
                        if k == 'length':
                            continue
                        accum[k] /= accum['length']

                    print("[%s] Epoch: %d, Step: %d, Loss: %f, mse: %f, entropy: %f"%(str(datetime.now()), epoch, self.global_step, accum['loss'],accum['mse'],accum['entropy']))
                    accum = defaultdict(float)

                del(output)
                self.global_step += 1

        else:
            for step, input_pack in enumerate(self.inputdata):
                if self.global_step % self.opts['val_freq'] == 0:
                        self.validate()
        #self.model.train()
                        self.save_checkpoint(epoch)
             
                pred_tensor =self.preddata[step]
                input_batch2 = self.inputdata2[step]
            # Forward pass
                output = self.model(input_pack.cuda(),input_batch2,"train")
                loss_fn1 = nn.MSELoss()
                loss_fn2 = nn.CrossEntropyLoss()
      #	print output[:,:-2]
                mse = loss_fn1(output[:,:-2],pred_tensor[:,:-2])
                output1 = output[:,-1].view(-1,1)
                pred = pred_tensor[:,-1]
        #print 'mse %f'%mse
    #entropy = loss_fn2(torch.cat((1-output1,output1),dim=1),pred.type(torch.cuda.LongTensor))
                entropy = loss_fn1(output[:,-1],pred)		
                loss = 10*mse+entropy
    # Backward 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                accum['loss'] += float(loss)
                accum['mse'] +=float(mse)
                accum['entropy']+=float(entropy)
                accum['length'] += 1
                if(step%self.opts['print_freq']==0):
                # Mean of accumulated values
                    for k in accum.keys():
                        if k == 'length':
                            continue
                        accum[k] /= accum['length']

                    print("[%s] Epoch: %d, Step: %d, Loss: %f ,mse: %f ,entropy %f"%(str(datetime.now()), epoch, self.global_step, accum['loss'],accum['mse'],accum['entropy']))
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
        count = [0]*18
        with torch.no_grad():
            if self.key==0:
                self.key+=1
                for step, data in enumerate(tqdm(self.val_loader)):
                    input_batch,input_batch2,pred_batch,x0s,y0s,ws,hs,dec_per = self.change_data(data)
                    input_pack = rnn.pack_sequence(input_batch)
                    input_batch2= torch.cat(input_batch2)
                    input_batch2 = input_batch2.to(device)
                    self.valdata2.append(input_batch2)
                    self.valdata.append(input_batch)
                    self.dec_per_val.append(dec_per)
                    output= self.model(input_pack.cuda(),input_batch2,"test")
                    output = output.cpu()

                    pred_batch = torch.cat(pred_batch)
                    self.predvaldata.append(pred_batch)
                    list = []
                    list.append(input_batch[0])
                    list.append(output)
                    output = torch.cat(list)
                    list = []
                    list.append(input_batch[0])
                    list.append(pred_batch)
                    pred = torch.cat(list)
                    self.valx.append(x0s)
                    self.valy.append(y0s)
                    self.valw.append(ws)
                    self.valh.append(hs)
                    output = output.numpy()
                    output = output[:,:-2]
                    pred = pred.numpy()
                    pred = pred[:,:-2]
                    tmp = [0,0]
                        
                    for i in output:
                        i[0] = i[0]*ws[0]
                        i[1] = i[1]*hs[0]
                        i[0] += tmp[0]
                        i[1] +=tmp[1]
                        if i[0]<=0:
                            i[0] = 0
                        if i[1]<=0:
                            i[1] = 0
                        if i[0]>ws[0]:
                            i[0] = ws[0]
                        if i[1]>hs[0]:
                            i[1] = hs[0]
                        tmp = i
                    tmp = [0,0]
                    for i in pred:
                        i[0] = i[0]*ws[0]
                        i[1] = i[1]*hs[0]
                        i[0]+=tmp[0]
                        i[1]+=tmp[1]
                        tmp = i
                    output = output.astype(int)
                    pred = pred.astype(int)
                    iou = iou_from_poly(output,pred,ws[0],hs[0])
                    ious[int((dec_per[0]*100)/5-1)]+=iou
                    count[int((dec_per[0]*100)/5-1)]+=1
                        #print 'sucessful'
                    del(output)
                avg_loss = [0]*18
                for i in range(len(ious)):
                    avg_loss[i] = float(ious[i]/count[i])
                    print("avg loss for %f dec_per is %f"%((i+1)*0.05,avg_loss[i]))

            else:
                visited = 0
                for step,input_batch in enumerate(self.valdata):
                    pred_tensor = self.predvaldata[step]
                    dec_per = self.dec_per_val[step]
                    input_batch2 = self.valdata2[step]
                    input_pack = rnn.pack_sequence(input_batch)
                    x0s = self.valx[step]
                    y0s = self.valy[step]
                    ws = self.valw[step]
                    hs = self.valh[step]
                    output = self.model(input_pack.cuda(),input_batch2,"test")
                #if self.epoch>=25:
                #print output
                    output = output.cpu()
                    list = []
                    list.append(input_batch[0])
                    list.append(output)
                    output = torch.cat(list)
                    list = []
                    list.append(input_batch[0])
                    list.append(pred_tensor)
                    pred = torch.cat(list)
                    output = output.numpy()
                    output = output[:,:-2]
                    pred = pred.numpy()
                    pred = pred[:,:-2]
                    tmp = [0,0]
                    for i in output:
                        i[0] = i[0]*ws[0]
                        i[1] = i[1]*hs[0]
                        i[0] += tmp[0]
                        i[1] +=tmp[1]
                        if i[0]<=0:
                            i[0] =0
                        if i[1] <=0:
                            i[1]=0
                        if i[0]>ws[0]:
                            i[0] =ws[0]
                        if i[1]>hs[0]:
                            i[1] = hs[0]
                        tmp = i
                    tmp = [0,0]
                    for i in pred:
                        i[0] = i[0]*ws[0]
                        i[1] = i[1]*hs[0]
                        i[0]+=tmp[0]
                        i[1]+=tmp[1]
                        tmp = i
                    output = output.astype(int)
                    pred = pred.astype(int)
                    iou = iou_from_poly(output,pred,ws[0],hs[0])
                #print 'decper is %f'%dec_per[0]
                #if dec_per[0]==0.6:
#print 'output_len'
#print len(output)
#print 'pred_len'
#print len(pred)
#print 'iou is %f'%iou
                    ious[int((dec_per[0]*100)/5-1)]+=iou
                    count[int((dec_per[0]*100)/5-1)]+=1
                    del(output)
                avg_loss = [0]*18
                for i in range(len(ious)):
                    avg_loss[i] = ious[i]/count[i]
                    print("avg ious for %f dec_per is %f"%((i+1)*0.05,avg_loss[i]))
                self.key+=1
        self.model.train()
if __name__ == '__main__':
    args = get_args()
    opts = json.load(open(args.exp, 'r'))
    for dec_per in range(5,95,5):
        opts["dec_per"]=float(dec_per)/100
        opts["dataset"]["train"]["dec_per"] = float(dec_per)/100
        opts['max_epochs'] = 40
        print(opts)
        trainer = Trainer(args,opts)
        trainer.loop()
