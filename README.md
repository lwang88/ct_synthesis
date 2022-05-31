# cvpr_ct_synthesis
## Introduction:
This is the code of paper named " [Incremental Cross-view Mutual Distillation for Self-supervised Medical CT Synthesis](https://arxiv.org/abs/2112.10325)". Considering that the ground-truth intermediate medical slices are always absent in clinical practice, we introduce the incremental cross-view mutual distillation strategy to accomplish this task in the self-supervised learning manner. Specifically, we model this problem from three different views: slice-wise interpolation from axial view and pixel-wise interpolation from coronal and sagittal views. Under this circumstance, the models learned from different views can distill valuable knowledge to guide the learning processes of each other. We can repeat this process to make the models synthesize intermediate slice data with increasing inter-slice resolution.

## Requirement:
PyTorch>=0.4.1       
nibabel   
pytorch_wavelets   

## Examples:
git clone https://github.com/wangliang88/cvpr_ct_synthesis.git   
cd cvpr_ct_synthesis   
python main.py --upscale 2 --batch_size 4 --lr 1e-5 --data_dir <path of the train data>

The data directory should follow the pattern below:   
|-traindir   
   |--volume1.nii.gz   
   |--volume2.nii.gz   
   |--volume3.nii.gz   
   |--...   
|-testdir   
  |--volume1.nii.gz   
  |--volume2.nii.gz   
  |--volume3.nii.gz   
  |--...  

## Citation: 
If you find this work or code is helpful in your research, please cite:   
 @article{wangliang,   
 title={Incremental Cross-view Mutual Distillation for Self-supervised Medical CT Synthesis },   
 author={Chaowei Fang, Liang Wang, Dingwen Zhang,Jun Xu,Yixuan Yuan,Junwei Han},   
 booktitle={CVPR},   
 year={2022}   
}
