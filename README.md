# MPSTN：Leveraging Intra-Period and Inter-Period Features for Enhanced Urban Rail Passenger Flow Prediction
Code for paper 'Leveraging Intra-Period and Inter-Period Features for Enhanced Urban Rail Passenger Flow Prediction'.<br /> Arxiv link for the paper will be give soon.<br />
Main insight for the paper is folding 1-d time-serise data as matrix according to the period. Then using CNN to dig the intra and inter period information and using GNN to intergrate spatial information

## Train
### Train Model for Each Station
python cnn_mian.py

### Train Model for the Whole Subway System
python cnn_gnn2.py

## Requirements
python 3.7 <br />
torch  1.13.0+cu117
