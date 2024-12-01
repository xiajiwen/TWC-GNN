# TWC-GNN

 TWC-GNN uses node degrees to define higher-order topological structures, assesses node importance, and captures mutual interactions between central nodes and their adjacent counterparts. This approach improves our understanding of complex relationships within the network.



## Setup

For a quick setup follow the next steps:

conda create -n TWCGNN python=3.8

conda activate TWCGNN 

git clone https://github.com/xiajiwen/TWC-GNN

pip install -r requirements.txt



## Running

Put the corresponding data set in the corresponding file, then run train.py, and then run on the trained model to test, such as the reddit dataset run twgnn_reddit.py

