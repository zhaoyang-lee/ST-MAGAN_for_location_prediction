# ST-MAGAN_for_location_prediction
this is code for location prediction with ST-MAGAN

# MyModel:

[TOC]

# files
## data_analysis
1. Range filter
2. Remove the null and repeatation
3. Rank the data by user and time
4. Remove the trajectories whose length < 28
5. Analysis POI distribution

## node2vec
Apply skip-gram algrithm to get latent vectors of every POI and region

## preprocess
1. Add the "Region" column which represent the maphash string of every location.
2. Calation the distance of between every 2 region as the weight of "Region-Graph1".
3. Number the regions as the node of "Region-Graph1".
4. calculate the visit times between 2 regions in time_period as the weight of "Region-Graph2"
5. Split dataset to train and test data.
6. Calcute the feature vectors of Region-Graph1 and Region-Graph2
7. Save all data as corresponding files 


## graph_build
Build 2 graph class--"Graph1" and "Graph2". Graph1 stands for the "Region-Graph1" which weigh the geographic dependence of regions; and Graph2 represents the directed graph of user co-visit times, as well as "Region-Graph2". There are following functions in the two classes:

1. Load data from the part--preprocess
2. Load the distance matrix data, and transfer it to edge weight of Region-Graph1. Transfer the matrix to node pairs and weights. Remember to transfer data to torch.
3. Load the covisit matrix data, and transfer it to edge weight of Region-Graph2. Transfer the matrix to node pairs and weights. Remember to transfer data to torch.
4. Build Region-Graph1 and Region-Graph2.(edges, nodes, weights, node feature vectors)

## STLSTM
Add time factor and spatial factor into LSTM.

## ST-MAGCN model


# run
First, run "preprocess.ipynb" get all data we need.

Then run "ST-MAGCN model".
