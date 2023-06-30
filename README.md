# Federated Route Forecasting over Maritime Data Silos (FedVRF)
### Official Python implementation of the FedVRF model from the paper "Federated Vessel Route Forecasting over Maritime Data Silos"


# Installation 

In order to use FedVRF in your project, download all necessary modules in your directory of choice via pip or conda, and install their corresponding dependencies, as the following commands suggest:

```Python
# Using pip/virtualenv
pip install −r requirements.txt

# Using conda
conda install --file requirements.txt
```


# Usage

In order to train local VRF instance in SN-CML (SD-CML, respectively), please run the script ```training-rnn-sncml.py``` (```training-rnn-sdcml.py```, respectively). In order to train FedVRF, please run the ```aggregation-server.py``` script, and the ```fedvrf-client.py``` script for as many available datasets. 

In order to reproduce the experimental study of the paper, please run the code in the notebooks located in the ```experimental-study``` directory. To load and preprocess the dataset(s), please run the appropriate code in [https://github.com/DataStories-UniPi/VLF_VRF](https://github.com/DataStories-UniPi/VLF_VRF).


# Contributors
Andreas Tritsarolis; Department of Informatics, University of Piraeus

Nikos Pelekis; Department of Statistics & Insurance Science, University of Piraeus

Konstantina Bereta; MarineTraffic

Dimitris Zissis; Department of Product & Systems Design Engineering, University of the Aegean & MarineTraffic

Yannis Theodoridis; Department of Informatics, University of Piraeus
