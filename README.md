# Neural-Inpainting-Forschungszentrum-Juelich

Code for the Master Thesis: Neural Inpainting: Repairing Artifacts in Histological Brain Sections with 
Deep Generative Models.

Download the thesis here:

Small version (31MB, without examples in the appendix): https://uni-duesseldorf.sciebo.de/s/B9tD5vpFFek8faZ, password: MTdownload

Full version (159MB): https://uni-duesseldorf.sciebo.de/s/uWG01Zh1RM0av0J, password: MTdownload

![alt text](https://github.com/KaiserTim/Neural-Inpainting-Repairing-Histological-Artifacts/blob/master/utils/NN_overview.png?raw=true)

## File Structure
The intact dataset and cGlow model saves couldn't be uploaded to GitHub, due to filesize restrictions.

Download here: https://uni-duesseldorf.sciebo.de/s/ISTlJJHUCzPstNt, password: NIdownload

### Demo
A notebook to test all models by running inference examples. 

### models
Contains scripts for running each model, the model classes and model saves. Each script contains class that loads the model when initialized, as well as
an inference function to run it. 

### data
Contains both datasets in TIF files.

### training
Contains the necessary code to train and evaluate each model, as well as a model save.

### utils
Contains code for the model evaluations, dataset implementations and utility functions.
