from torch.utils.data import Dataset,DataLoader
import numpy as np
import itertools
import torch
class Dataset_as_cv(Dataset):
    '''
    dataset for each station,return a matrix
    data:np.array,shape as (day_num,time_num,station_num),for only inflow
    return: 
        x1: tensor (seq_day,time_num)
        y:tensor (4) the inflow of future 4 time
    '''
    def __init__(self,data,seq_day):
        super(Dataset_as_cv).__init__()
        self.data=data
        self.seq_day=seq_day
        self.station_num=data.shape[2]
        self.day_num,self.time_num=data.shape[0],data.shape[1]
    def __len__(self):
        return (self.day_num-self.seq_day-1)*(self.time_num-8)*self.station_num
    
    def __getitem__(self, index):
        pred_day=index//((self.time_num-8)*self.station_num)
        station=(index-pred_day*((self.time_num-8)*self.station_num))//(self.time_num-8)
        pred_time=index-pred_day*((self.time_num-8)*self.station_num)-(self.time_num-8)*station
        x1=self.data[pred_day:pred_day+self.seq_day+1,:,station].copy()
        # for i in range(pred_time+4,self.time_num):
        #     x1[-1,i]=0
        x1=x1.reshape(-1)[pred_time+4:self.seq_day*(self.time_num)+pred_time+4].reshape(self.seq_day,(self.time_num))
        y=self.data[pred_day+self.seq_day,pred_time+4:pred_time+8,station]
        return torch.tensor(x1,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)
    
class Dataset_stations(Dataset):
    '''
    dataset for each station,return a matrix
    data:np.array,shape as (day_num,time_num,station_num,2)
    return: 
        x1: tensor (4,station,2)
        y:tensor (station,2,4) the flow of future 4 time
    '''
    def __init__(self,data):
        super(Dataset_stations).__init__()
        self.data=data
        self.station_num=data.shape[2]
        self.day_num,self.time_num=data.shape[0],data.shape[1]
    def __len__(self):
        return (self.day_num)*(self.time_num-8)
    def __getitem__(self, index):
        pred_day=index//(self.time_num-8)
        pred_time=index-(self.time_num-8)*pred_day
        x=self.data[pred_day,pred_time:pred_time+4].copy()
        y=self.data[pred_day,pred_time+4:pred_time+8].copy()
        return torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)
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
        x=x.reshape(15*73,288*2)[pred_time+4:self.seq_day*(self.time_num)+pred_time+4].reshape(14,73,288,2)##14*73*288*2
        y=self.data[pred_day+14,pred_time+4:pred_time+8].copy().reshape(4,-1)  ##4*288*2->4,576
        return torch.tensor(x,dtype=torch.float32).permute(2,3,0,1),torch.tensor(self.weather[pred_day+14],dtype=torch.long),torch.tensor(y,dtype=torch.float32).permute(1,0)

class Dataset_as_cv2(Dataset):
    '''
    dataset for each station,return a matrix
    data:np.array,shape as (day_num,time_num,station_num,2),for only inflow and outflow
    return: 
        x1: tensor (seq_day,time_num)
        y:tensor (4,2) the inflow of future 4 time
    '''
    def __init__(self,data,seq_day):
        super(Dataset_as_cv).__init__()
        self.data=data
        self.seq_day=seq_day
        self.station_num=data.shape[2]
        self.day_num,self.time_num=data.shape[0],data.shape[1]
    def __len__(self):
        return (self.day_num-self.seq_day-1)*(self.time_num-8)*self.station_num
    
    def __getitem__(self, index):
        pred_day=index//((self.time_num-8)*self.station_num)
        station=(index-pred_day*((self.time_num-8)*self.station_num))//(self.time_num-8)
        pred_time=index-pred_day*((self.time_num-8)*self.station_num)-(self.time_num-8)*station
        x1=self.data[pred_day:pred_day+self.seq_day+1,:,station,:].copy()
        # for i in range(pred_time+4,self.time_num):
        #     x1[-1,i]=0
        x=x1.reshape(-1,2)[pred_time:self.seq_day*(self.time_num)+pred_time].reshape(self.seq_day,(self.time_num),2).transpose(2,0,1)
        q=x1.reshape(-1,2)[self.seq_day*(self.time_num)+pred_time:self.seq_day*(self.time_num)+pred_time+4].reshape(-1)
        y=self.data[pred_day+self.seq_day,pred_time+4:pred_time+8,station]
        return torch.tensor(x,dtype=torch.float32),torch.tensor(q,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)
    
class Dataset_system(Dataset):
    '''
    dataset for each station,return a matrix
    data:np.array,shape as (day_num,time_num,station_num,2)
    return: 
        x: tensor (288,2,14,73)
        q: (288,2,4)
        y:tensor (station,2,4) the flow of future 4 time
    '''
    def __init__(self,data):
        super(Dataset_system).__init__()
        self.data=data
        self.station_num=data.shape[2]
        self.day_num,self.time_num=data.shape[0],data.shape[1]
        self.seq_day=14
    def __len__(self):
        return (self.day_num-15)*(self.time_num-8)
    def __getitem__(self, index):
        pred_day=index//(self.time_num-8)
        pred_time=index-(self.time_num-8)*pred_day
        x=self.data[pred_day:pred_day+self.seq_day+1,:,:].copy() 
        x1=x.reshape(15*73,288*2)[pred_time:self.seq_day*(self.time_num)+pred_time].reshape(14,73,288,2).transpose(2,3,0,1)   
        q=x.reshape(15*73,288*2)[self.seq_day*(self.time_num)+pred_time:self.seq_day*(self.time_num)+pred_time+4].reshape(4,288,2).transpose(1,2,0)
        y=self.data[pred_day+14,pred_time+4:pred_time+8].transpose(1,2,0)  
        return torch.tensor(x1,dtype=torch.float32),torch.tensor(q,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)