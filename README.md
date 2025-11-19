# Concept-Bottleneck-Models

## The GitHub repository of the concept bottleneck model is in https://github.com/yewsiang/ConceptBottleneck.

Make the conda environment with the name of bottleneck

activate conda bottleneck

cd to ~/ConceptBottleneck/src

pip install -r requirements.txt

error:
ERROR: Could not find a version that satisfies the requirement en-core-web-sm==2.1.0 (from versions: none)
ERROR: No matching distribution found for en-core-web-sm==2.1.0

solution:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz

then run the requirement again

error:
ERROR: Could not find a version that satisfies the requirement pygobject==3.20.0 (from versions: 3.27.0, 3.27.1, 3.27.2, 3.27.3, 3.27.4, 3.27.5, 3.28.0, 3.28.1, 3.28.2, 3.28.3, 3.29.1.dev0, 3.29.2.dev0, 3.29.3.dev0, 3.30.0, 3.30.1, 3.30.2, 3.30.3, 3.30.4, 3.30.5, 3.31.1.dev0, 3.31.2.dev0, 3.31.3.dev0, 3.31.4.dev0, 3.32.0, 3.32.1, 3.32.2, 3.33.1.dev0, 3.34.0, 3.36.0, 3.36.1, 3.38.0, 3.40.0, 3.40.1, 3.42.0, 3.42.1, 3.42.2, 3.43.1.dev0, 3.44.0, 3.44.1)
ERROR: No matching distribution found for pygobject==3.20.0

solution: 
sudo apt install libgirepository1.0-dev
pip3 install pygobject
remove the version of pygobject 
remove the version of python-apt 
it doesn not work I totally remove the python-apt
remove torch version torch
torchvision
grpcio
numpy
pandas
python-dateutil
PyWavelets

pip install scikit-learn

RuntimeError: No CUDA GPUs are available
https://www.nvidia.com/Download/index.aspx
choose : NVIDIA ® GeForce ® GTX 1650 GPU (This works for my computer)
download
sudo apt install NVIDIA-Linux-x86_64-535.104.05.run
error: unable to locate
sudo apt-get install nvidia-driver
then
sudo bash NVIDIA-Linux-x86_64-535.104.05.run

https://www.tensorflow.org/install/pip
nvidia-smi

show the driver and it is not empty
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
it is incompatible with torch 2….

so
pip install nvidia-cudnn-cu11==8.5.0.96

python3 src/experiments.py cub Concept_XtoC --seed 1 -ckpt 1 -log_dir ConceptModel__Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir /home/hengam2/DeepLearning/project1/ConceptBottleneck/src/CUB_processed/class_attr_data_10 -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck

python3 src/experiments.py cub Joint --seed 1 -ckpt 1 -log_dir Joint0.01Model__Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir /home/hengam2/DeepLearning/project1/ConceptBottleneck/src/CUB_processed/class_attr_data_10 -n_attributes 112 -normalize_loss -b 32 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -end2end

change the location of CUB_processed and CUB_200_2011

conda install pytorch torchvision cudatoolkit -c pytorch

with error of memory in GPU:
change the batch size from 64 to 32

to run validation of table1 with pretraied downloaded models:
Change the view in src/analysis.py with reshape

run code for independent model: 

python3 src/CUB/inference.py 
-model_dirs pretrained_NN_coda/Concept/ConceptModel__Seed1/outputs/best_model_1.pth pretrained_NN_coda/Concept/ConceptModel__Seed2/outputs/best_model_2.pth pretrained_NN_coda/Concept/ConceptModel__Seed3/outputs/best_model_3.pth -model_dirs2 pretrained_NN_coda/Independent/IndependentModel_WithVal___Seed1/outputs/best_model_1.pth pretrained_NN_coda/Independent/IndependentModel_WithVal___Seed2/outputs/best_model_2.pth pretrained_NN_coda/Independent/IndependentModel_WithVal___Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir src/CUB_processed/class_attr_data_10 -bottleneck -use_sigmoid -log_dir pretrained_NN_coda/Independent/IndependentModel__WithValSigmoid

All variable related to the accuracy in train test are saved in pretrained_NN_coda/Independent/IndependentModel__WithValSigmoid     /stdout   /result.txt


run code for Sequential model: 
python3 src/CUB/inference.py 
-model_dirs 
pretrained_NN_coda/Concept/ConceptModel__Seed1/outputs/best_model_1.pth 
pretrained_NN_coda/Concept/ConceptModel__Seed2/outputs/best_model_2.pth 
pretrained_NN_coda/Concept/ConceptModel__Seed3/outputs/best_model_3.pth 
-model_dirs2 
pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed1/outputs/best_model_1.pth pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed2/outputs/best_model_2.pth pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed3/outputs/best_model_3.pth 
-eval_data test -use_attr -n_attributes 112 
-data_dir src/CUB_processed/class_attr_data_10 
-bottleneck -feature_group_results 
-log_dir 
pretrained_NN_coda/Sequential/SequentialModel__WithVal             /stdout  /result.txt

run code for Joint model: 
python3 src/CUB/inference.py 
-model_dirs 
pretrained_NN_coda/Joint_0.001/Joint0.001Model_Seed1/outputs/best_model_1.pth pretrained_NN_coda/Joint_0.001/Joint0.001Model_Seed2/outputs/best_model_2.pth pretrained_NN_coda/Joint_0.001/Joint0.001Model_Seed3/outputs/best_model_3.pth 
-eval_data test -use_attr -n_attributes 112 
-data_dir src/CUB_processed/class_attr_data_10 
-log_dir 
pretrained_NN_coda/Joint_0.001/Joint0.001Model       /stdout  /result.txt

python3 src/CUB/inference.py -model_dirs pretrained_NN_coda/Joint_0.01/Joint0.01Model_Seed1/outputs/best_model_1.pth pretrained_NN_coda/Joint_0.01/Joint0.01Model_Seed2/outputs/best_model_2.pth pretrained_NN_coda/Joint_0.01/Joint0.01Model_Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir src/CUB_processed/class_attr_data_10 -log_dir pretrained_NN_coda/Joint_0.01/Joint0.01Model

python3 src/CUB/inference.py -model_dirs /home/hengam2/DeepLearning/project1/ConceptBottleneck/pretrained_NN_coda/Joint_0.01/Joint0.01Model__Seed1/outputs/best_model_1.pth /home/hengam2/DeepLearning/project1/ConceptBottleneck/pretrained_NN_coda/Joint_0.01/Joint0.01Model__Seed1/outputs/best_model_2.pth /home/hengam2/DeepLearning/project1/ConceptBottleneck/pretrained_NN_coda/Joint_0.01/Joint0.01Model__Seed1/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir src/CUB_processed/class_attr_data_10 -log_dir pretrained_NN_coda/Joint_0.01/Joint0.01Model 

run code for Standard model: 
python3 src/CUB/inference.py -model_dirs pretrained_NN_coda/Standard/Joint0Model_Seed1/outputs/best_model_1.pth pretrained_NN_coda/Standard/Joint0Model_Seed2/outputs/best_model_2.pth pretrained_NN_coda/Standard/Joint0Model_Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir src/CUB_processed/class_attr_data_10 -log_dir pretrained_NN_coda/Standard/Joint0Model                   /stdout  /result.txt

create new pkl file from the test data as a test_new and run each model with this new pkl file and with new log_file

python3 src/CUB/inference.py -model_dirs pretrained_NN_coda/Standard/Joint0Model_Seed1/outputs/best_model_1.pth pretrained_NN_coda/Standard/Joint0Model_Seed2/outputs/best_model_2.pth pretrained_NN_coda/Standard/Joint0Model_Seed3/outputs/best_model_3.pth -eval_data test_new -use_attr -n_attributes 112 -data_dir src/CUB_processed/class_attr_data_10 -log_dir pretrained_NN_coda/Standard/Joint0Model_new 


python3 src/CUB/tti.py -model_dirs 
-use_attr -mode random -n_trials 5 -use_invisible -class_level -data_dir2 /home/hengam2/DeepLearning/project1/ConceptBottleneck/pretrained_NN_coda/Joint_0.01/Joint0.01Model__Seed1/outputs/ best_model_1.pth /home/hengam2/DeepLearning/project1/ConceptBottleneck/pretrained_NN_coda/Joint_0.01/Joint0.01Model__Seed1/outputs/ best_model_2.pth -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 1



python3 src/CUB/inference.py 
python3 src/CUB/inference.py -model_dirs pretrained_NN_coda/Concept/ConceptModel__Seed1/outputs/best_model_1.pth pretrained_NN_coda/Concept/ConceptModel__Seed2/outputs/best_model_2.pth pretrained_NN_coda/Concept/ConceptModel__Seed3/outputs/best_model_3.pth -model_dirs2 pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed1/outputs/best_model_1.pth pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed2/outputs/best_model_2.pth pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir src/CUB_processed/class_attr_data_10 -bottleneck -feature_group_results -log_dir pretrained_NN_coda/Sequential/SequentialModel__WithVal_new

python3 src/CUB/tti.py -model_dirs pretrained_NN_coda/Joint_0.01/Joint0.01Model__Seed1/outputs/best_model_1.pth -use_attr -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 1

python3 src/CUB/tti.py -model_dirs pretrained_NN_coda/ConceptModel__Seed1/outputs/best_model_1.pth pretrained_NN_coda/ConceptModel__Seed2/outputs/best_model_2.pth pretrained_NN_coda/ConceptModel__Seed3/outputs/best_model_3.pth -model_dirs2 pretrained_NN_coda/SequentialModel_WithVal__Seed1/outputs/best_model_1.pth pretrained_NN_coda/SequentialModel_WithVal__Seed2/outputs/best_model_2.pth pretrained_NN_coda/SequentialModel_WithVal__Seed3/outputs/best_model_3.pth -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 1



python3 src/CUB/tti.py -model_dirs pretrained_NN_coda/Joint_0.01/Joint0.01Model__Seed1/outputs/best_model_1.pth pretrained_NN_coda/Joint_0.01/Joint0.01Model__Seed2/outputs/best_model_2.pth -use_attr -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 1

python3 src/CUB/tti.py -model_dirs pretrained_NN_coda/Concept/ConceptModel__Seed1/outputs/best_model_1.pth pretrained_NN_coda/Concept/ConceptModel__Seed2/outputs/best_model_2.pth pretrained_NN_coda/Concept/ConceptModel__Seed3/outputs/best_model_3.pth -model_dirs2 pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed1/outputs/best_model_1.pth pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed2/outputs/best_model_2.pth pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed3/outputs/best_model_3.pth -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 1

python3 src/CUB/tti.py -model_dirs pretrained_NN_coda/Concept/ConceptModel__Seed1/outputs/best_model_1.pth pretrained_NN_coda/Concept/ConceptModel__Seed2/outputs/best_model_2.pth pretrained_NN_coda/Concept/ConceptModel__Seed3/outputs/best_model_3.pth -model_dirs2 pretrained_NN_coda/Independent/IndependentModel_WithVal___Seed1/outputs/best_model_1.pth pretrained_NN_coda/Independent/IndependentModel_WithVal___Seed2/outputs/best_model_2.pth pretrained_NN_coda/Independent/IndependentModel_WithVal___Seed3/outputs/best_model_3.pth -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 1


python3 src/CUB/tti.py -model_dirs pretrained_NN_coda/Joint_0.01/Joint0.01Model__Seed2/outputs/best_model_2.pth -use_attr -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 1

python3 src/CUB/tti.py -model_dirs pretrained_NN_coda/Concept/ConceptModel__Seed3/outputs/best_model_3.pth -model_dirs2 pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed1/outputs/best_model_1.pth pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed2/outputs/best_model_2.pth pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed3/outputs/best_model_3.pth -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 1

python3 src/CUB/tti.py -model_dirs pretrained_NN_coda/Concept/ConceptModel__Seed1/outputs/best_model_1.pth -model_dirs2 pretrained_NN_coda/Independent/IndependentModel_WithVal___Seed2/outputs/best_model_2.pth  -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 1


python3 src/CUB/tti.py -model_dirs pretrained_NN_coda/Concept/ConceptModel__Seed2/outputs/best_model_2.pth -model_dirs2 pretrained_NN_coda/Independent/IndependentModel_WithVal___Seed3/outputs/best_model_3.pth -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 28

python3 src/CUB/tti.py -model_dirs pretrained_NN_coda/Concept/ConceptModel__Seed3/outputs/best_model_3.pth -model_dirs2 pretrained_NN_coda/Independent/IndependentModel_WithVal___Seed1/outputs/best_model_1.pth pretrained_NN_coda/Independent/IndependentModel_WithVal___Seed2/outputs/best_model_2.pth pretrained_NN_coda/Independent/IndependentModel_WithVal___Seed3/outputs/best_model_3.pth -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 28

python3 src/CUB/tti.py -model_dirs pretrained_NN_coda/Concept/ConceptModel__Seed3/outputs/best_model_3.pth -model_dirs2 pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed1/outputs/best_model_1.pth pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed2/outputs/best_model_2.pth pretrained_NN_coda/Sequential/SequentialModel_WithVal__Seed3/outputs/best_model_3.pth -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 28

python3 src/CUB/tti.py -model_dirs pretrained_NN_coda/Concept/ConceptModel__Seed3/outputs/best_model_3.pth -model_dirs2 pretrained_NN_coda/Independent/IndependentModel_WithVal___Seed3/outputs/best_model_3.pth -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 0

python3 src/CUB/tti.py -model_dirs pretrained_NN_coda/Joint_0.01/Joint0.01Model__Seed1/outputs/best_model_1.pth -use_attr -mode random -n_trials 5 -use_invisible -class_level -data_dir2 src/CUB_processed/class_attr_data_10 -data_dir src/CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model -n_attributes 112 -n_groups 28
