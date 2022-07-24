# ct_synthesis
## Introduction:
This is the code of paper named " [Incremental Cross view Mutual Distillation for Self supervised Medical CT Synthesis](https://arxiv.org/abs/2112.10325)". Considering that the ground-truth intermediate medical slices are always absent in clinical practice, we introduce the incremental cross-view mutual distillation strategy to accomplish this task in the self-supervised learning manner. Specifically, we model this problem from three different views: slice-wise interpolation from axial view and pixel-wise interpolation from coronal and sagittal views. Under this circumstance, the models learned from different views can distill valuable knowledge to guide the learning processes of each other. We can repeat this process to make the models synthesize intermediate slice data with increasing inter-slice resolution.

## Requirement:
cuda==10.1  
python==3.7  
PyTorch>=0.4.1       
nibabel   
pytorch_wavelets 
Some basic python packages such as Numpy, cv2, Scipy ......

## Examples:
git clone https://github.com/wangliang88/ct_synthesis.git   
cd cvpr_ct_synthesis   
python main.py --upscale 2 --batch_size 4 --lr 1e-5 --data_dir < path of the train data >   

The data directory should follow the pattern below:   
|-traindir   
&#160;&#160;&#160;|--volume1.nii.gz   
&#160;&#160;&#160;|--volume2.nii.gz   
&#160;&#160;&#160;|--volume3.nii.gz   
&#160;&#160;&#160;|--...   
|-testdir   
&#160;&#160;&#160;|--volume1.nii.gz   
&#160;&#160;&#160;|--volume2.nii.gz   
&#160;&#160;&#160;|--volume3.nii.gz   
&#160;&#160;&#160;|--... 

The slice number of each ct volume shoule be more than 33,if you use a ct volume which slices is less than 33,please modify the main.py "random.randint()".
