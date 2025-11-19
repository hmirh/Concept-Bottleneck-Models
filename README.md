# Concept-Bottleneck-Models

## Code description:

he code provided on the Concept bottleneck model GitHub contains numerous issues which make
it difficult to run and duplicate the results in paper. The GitHub repository of the concept bottleneck
model is in https://github.com/yewsiang/ConceptBottleneck. This project needs a list of packages.
To begin with, a Conda environment is created to install these main packages. Based on the instruction,
pip install -r requirements.txt should be run which contains all the required packages. However,
a bunch of errors appeared that are related to the mismatched versions of packages. Solving one error
leads to the appearance of another error. Due to this fact, each of these packages is installed separately
with pip install package name. At first, it tries to install the exact version of those packages that
are mentioned in the requirement.txt. Unless there is an error during installation, the version number
is removed to automatically install the package version that is compatible with other installed packages.
During package installations, the following error also appears:
ERROR: Could not find a version that satisfies the requirement en-core-web-sm==2.1.0
ERROR: No matching distribution found for en-core-web-sm==2.1.0
The solution is:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web
_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz}
To run the codes mentioned in this repository, there are some prerequisite files, such as the official
CUB dataset (CUB 200 2011), processed CUB data (CUB processed), places365 dataset (places365),
and pretrained Inception V3 models (pretrained) that should be downloaded from the Codalab work-
sheet. All these data should be placed in src folder.
To train the pretrained Inception V3 on the CUB dataset the experiments.py should be run. When
calling the experiments.py, data dir is the directory of the CUB processed folder, and log dir is the
directory to save the trained model. The following command shows how to call experiments.py.
python3 src/experiments.py cub Joint --seed 1 -ckpt 1
-log_dir Joint0.01Model__Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux
-use_attr -weighted_loss multiple -data_dir /home/hengam2/DeepLearning/project1/
ConceptBottleneck/src/ CUB_processed/class_attr_data_10 -n_attributes 112
-normalize_loss -b 32 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -end2end
By running the above code, the error of RuntimeError: No CUDA GPUs are available appeared,
thus the environment was set up to run on GPU. After that, the error of torch.cuda.OutOfMemoryError:
CUDA out of memory appeared. This issue is solved by changing the batch size from 64 to 32. This pro-
cess leads to training the pretrained Inception V3 over the CUB dataset to have the concept bottleneck
models. For replicating this process, the Joint concept bottleneck model is chosen to show the validity
of this process. This code is run successfully and trains the Joint model. However, the training process
takes more than 24 hours to reach an accuracy of 92.55.
To replicate the results of Table 1, it is required to download the trained 3 seeds of each model,
Independent, Sequential, and Joint models as well as the Standard model. Then, apply inference.py
to the test set.
python3 src/CUB/inference.py -model_dirs
pretrained_NN_coda/Joint_0.001/Joint0.001Model_Seed1/outputs/best_model_1.pth
1
pretrained_NN_coda/Joint_0.001/Joint0.001Model_Seed2/outputs/best_model_2.pth
pretrained_NN_coda/Joint_0.001/Joint0.001Model_Seed3/outputs/best_model_3.pth
eval_data test -use_attr -n_attributes 112
-data_dir src/CUB_processed/class_attr_data_10
-log_dir pretrained_NN_coda/Joint_0.001/Joint0
In this code, the model dirs specifies the location of trained models and the log dir specifies
the location of saving the result of the experiment. In this directory, there are two files, stdout and
result.txt, in which the experiment details are shown, containing the accuracy and error of concept
and target predictions. The following error has appeared when running the inference.py.
RuntimeError: view size is not compatible with input tensor’s size and stride (at
least one dimension spans across two contiguous subspaces).
It seems there is a version incompatibility issue. By replacing the .view() with a .reshape() instead,
this issue was solved. There is another error which is related to the indexing issue, the error is as follows:
Traceback (most recent call last):
File "src/CUB/tti.py", line 351, in <module>
values = run(args)
File "src/CUB/tti.py", line 233, in run
instance_attr_labels.extend(list(np.array(d[’attribute_label’])[mask]))
IndexError: index 112 is out of bounds for axis 0 with size
The solution is changing class _attr_count = np.zeros((N_CLASSES, N_ATTRIBUTES, 2)) to class_attr_count
= np.zeros((N_CLASSES, args.n_attributes, 2)) and adding -n_attributes 112 when calling tti.py.
To replicate Figure 3, the tti.py script should be applied to the two specified images. To find these
two images from datasets, run the following code.
import pickle
with open("src/CUB_processed/class_attr_data_10/test.pkl", "rb") as f:
content = pickle.load(f)
for index, item in enumerate(content):
if item[’class_label’] == 180: #59
print(index, item)
part_of_data = [content[1682], content[5225]]
output_filename = ’src/CUB_processed/class_attr_data_10/test_new.pkl’
## Open the new .pkl file for writing
with open(output_filename, ’wb’) as file:
pickle.dump(part_of_data, file)
Now, we have the list of images that have the class labels 59 and 180. We are trying to find those
specific images that are categorized in the Glaucous winged Gull and Worm eating Warbler classes.
Then, plot the image of this list to find those specific images.
In the downloaded CUB 200 2011 dataset, there is classes.txt which specifies the class name based
on the class labels. There is an inconsistency between the class labels used in this text file and the ones
used in the dataset. The class labels in classes.txt are defined in the range 1 to 200 while the class
labels in datasets are in the range 0 to 199. By considering this point the results obtained from tti.py
can be analyzed accurately.
It is complicated to figure out how tti.py script intervenes on concepts. Here, b attr new is a new
concept that is modified based on the method explained in the paper. Then, model2 predicts the class
labels with these new concepts. To call tti.py, the -data_dir shows the location of a new dataset,
-n_groups specifies the number of concepts that are intervened, it can change between 0 to 28. If the
number of concepts intervened is equal to zero, it means that no intervention is applied to the model
during the test time. The following code is an example of running tti.py on the new dataset with the
seed1 of the Joint model and -n_groups = 28.
2
python3 src/CUB/tti.py
-model_dirs
pretrained_NN_coda/Joint_0.01/Joint0.01Model__Seed1/outputs/best_model_1.pth
-use_attr -mode random -n_trials 5 -use_invisible -class_level
-data_dir2 src/CUB_processed/class_attr_data_10
-data_dir src/CUB_processed/class_attr_data_10
-log_dir TTI__Joint0.01Model
-n_attributes 112 -n_groups 28
In tti.py script, for changing the whole number of intervened concepts to the desired value, such as
0 or 1, the command line of b_attr_new = np.full_like(b_attr_new, 1) should be used.
