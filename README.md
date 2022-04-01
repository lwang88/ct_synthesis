# cvpr_ct_synthesis
## Introduction:
this is the code of paper named " Incremental Cross-view Mutual Distillation for Self-supervised Medical CT Synthesis",[paper](https://arxiv.org/abs/2112.10325)
    Considering that the ground-truth intermediate medical slices are always absent in clinical practice, we introduce the incremental cross-view mutual distillation strategy to accomplish this task in the self-supervised learning manner. Specifically, we model this problem from three different views:slice-wise interpolation from axial view and pixel-wise interpolation from coronal and sagittal views. Under this circumstance, the models learned from different views can distill valuable knowledge to guide the learning processes of
each other. We can repeat this process to make the models synthesize intermediate slice data with increasing interslice resolution. To demonstrate the effectiveness of the proposed approach, we conduct comprehensive experiments on
a large-scale CT dataset. Quantitative and qualitative comparison results show that our method outperforms state-ofthe-art algorithms by clear margins

## Requirement:
PyTorch>=0.4.1       
nibabel   
pytorch_wavelets   

## Examples:
git clone https://github.com/wangliang88/cvpr_ct_synthesis.git   
cd cvpr_ct_synthesis   
python main.py --upscale 2 --batch_size 4 --lr 1e-5 --data_dir <path of your train data>   

your datadir should follow the pattern below:   
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
 when your upscale is 2 ,the data should crop as w(256 or small)*l(256)*h(13 or 15 or 17 ...)   
 when your upscale is 3 ,the data should crop as w(128 or small)*l(128)*h(19 ...)   
 when your upscale is 4 ,the data should crop as w(64 or small)*l(64)*h(33 ...)   
    
## Citation:
If you find this work or code is helpful in your research, please cite:   
 @article{wangliang,   
 title={Incremental Cross-view Mutual Distillation for Self-supervised Medical CT Synthesis },   
 author={Chaowei Fang, Liang Wang, Dingwen Zhang,Jun Xu,Yixuan Yuan,Junwei Han},   
 booktitle={CVPR},   
 year={2022}   
}
