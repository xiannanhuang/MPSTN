import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import Model
import Dataset
import numpy as np
import logging
from datetime import datetime
from torch.optim.lr_scheduler import StepLR


dataset = Dataset.Dataset_stations_all(np.load('data\\sh_data.npy')[:62, :, :, :],np.load('data\\weather.npy')[:62])
validdataset = Dataset.Dataset_stations_all(np.load('data\\sh_data.npy')[48:69 ,:,:, :],np.load('data\\weather.npy')[48:69])

device = torch.device('cuda')
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
validdataloader = DataLoader(validdataset, batch_size=batch_size, shuffle=False)

log_file = f'log\\cnn_gnn_training_log_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt'
logging.basicConfig(filename=log_file, level=logging.INFO)
model=Model.MyCNNandGNN2().to(device)
# Define the loss function and optimizer
criterion = torch.nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = StepLR(optimizer, step_size=2, gamma=0.95)
# Train the model
num_epochs = 200

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logging.info(f'{timestamp} -' + 'train start for model cnn_trans_stations')
adj_matrix3=torch.tensor(np.load('data\\graph_sh_conn.pkl',allow_pickle=True),dtype=torch.float32).to(device)
# Record the names and shapes of the network's parameters
for name, param in model.named_parameters():
    logging.info(f"Parameter Name: {name}\t Shape: {param.shape}")

# Calculate and record the total number of trainable parameters
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Total Trainable Parameters: {total_trainable_params}")

# Record the current values of various parameters
logging.info(f"num_epochs: {num_epochs}")
logging.info(f"learning rate: {optimizer.param_groups[0]['lr']}")
best_valid_loss = float('inf')  # Initialize the minimum test loss as infinity
for epoch in range(num_epochs):
    epoch_start_time=datetime.now()
    model.train()
    running_loss = 0.0
    i = 0
    for inputs,weather, q,labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        weather=weather.to(device)
        q=q.to(device)
        # Clear the gradients of the optimizer
        optimizer.zero_grad()
        # Forward pass
        outputs = model.forward(inputs,adj_matrix3,weather,q)
        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and update model parameters
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        running_loss += loss.item()
        i += 1

        # Print training progress
        if (i + 1) % 100 ==0:
            avg_loss = running_loss / 100
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1,
                                                                          len(dataloader), avg_loss)
            print(log_msg)
            logging.info(f'{timestamp} - {log_msg}')
            running_loss = 0.0
    model.eval()
    with torch.no_grad():
        val_loss = torch.tensor([0.,0.,0.,0.]).to(device)
        for inputs,weather,q, labels in validdataloader:
            q=q.to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)
            weather=weather.to(device)
            outputs = model.forward(inputs,adj_matrix3,weather,q)
            loss = torch.abs(outputs-labels).mean(dim=[0,1,2])
            val_loss += loss
        avg_val_loss = val_loss / len(validdataloader)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = 'Validation Loss 15min: {:.4f};Validation Loss 30min: {:.4f};Validation Loss 45min: {:.4f};Validation Loss 60min: {:.4f}'.format(avg_val_loss[0],avg_val_loss[1],avg_val_loss[2],avg_val_loss[3])
        print(log_msg)
        logging.info(f'{timestamp} - {log_msg}')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch_duration = datetime.now() - epoch_start_time
        log_msg = f"Epoch: {epoch+1}\tLearning rate: {optimizer.param_groups[0]['lr']:.6f}\tDuration: {epoch_duration}"
        print(log_msg)
        logging.info(f"{timestamp} - {log_msg}")
        if avg_val_loss.sum() < best_valid_loss:
            best_test_loss = avg_test_loss.sum()
            best_model_name = f"model\\cnn_gnn_best_model_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth"
            torch.save(model.state_dict(), best_model_name)
            logging.info(f"Saved best model: {best_model_name}")
    scheduler.step()
# Generate a model name with the current time
model_name = f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth"

# Save the model
torch.save(model.state_dict(), model_name)

# Print the stored model name
logging.info(f"Saved final model: {model_name}")

