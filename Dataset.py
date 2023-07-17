from torch.utils.data import Dataset,DataLoader
import numpy as np
import itertools
import torch

class Dataset_stations_all(Dataset):
    '''
    dataset for each station,return a matrix
    data:np.array,shape as (day_num,time_num,station_num,2)
    return: 
        x: tensor (288*2,14,73)
        y:tensor (station,2,4) the flow of future 4 time
    '''
    def __init__(self,data,weather):
        super(Dataset_stations).__init__()
        self.data=data
        self.station_num=data.shape[2]
        self.day_num,self.time_num=data.shape[0],data.shape[1]
        self.seq_day=14
        self.weather=weather
    def __len__(self):
        return (self.day_num-15)*(self.time_num-8)
    def __getitem__(self, index):
        pred_day=index//(self.time_num-8)
        pred_time=index-(self.time_num-8)*pred_day
        x=self.data[pred_day:pred_day+self.seq_day+1,:,:].copy() ##15*73*288*2
        x1=x.reshape(15*73,288*2)[pred_time:self.seq_day*(self.time_num)+pred_time].reshape(14,73,288,2).transpose(2,3,0,1)   
        q=x.reshape(15*73,288*2)[self.seq_day*(self.time_num)+pred_time:self.seq_day*(self.time_num)+pred_time+4].reshape(4,288,2).transpose(1,2,0)
        y=self.data[pred_day+self.seq_day,pred_time+4:pred_time+8].transpose(1,2,0)  
        return torch.tensor(x1,dtype=torch.float32),torch.tensor(self.weather[pred_day+self.seq_day],dtype=torch.long),torch.tensor(q,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)
