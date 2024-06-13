# TBUnet
TBUnet: A Pure Convolutional U‑Net Capable of Multifaceted Feature Extraction for Medical Image Segmentation
Authors: LiFang Chen，Jiawei Li，Hongze Ge
<br/>
Research paper: https://doi.org/10.1007/s10916-023-02014-2

## Abstract
<div align="justify">
Many current medical image segmentation methods utilize convolutional neural networks (CNNs), with some extended 
U-Net-based networks relying on deep feature representations to achieve satisfactory results. However, due to the limited 
receptive fields of convolutional architectures, they are unable to explicitly model the varying range dependencies present 
in medical images. Recently, advancements in large kernel convolution have allowed for the extraction of a wider range of 
low frequency information, making this task more achievable. In this paper, we propose TBUnet for solving the problem of 
difficult to accurately segment lesions with heterogeneous structures and fuzzy borders, such as melanoma, colon polyps and 
breast cancer. The TBUnet is a pure convolutional network with three branches for extracting high frequency information, 
low frequency information, and boundary information, respectively. It is capable of extracting features in various areas. To 
fuse the feature maps from the three branches, TBUnet presents the FL (fusion layer) module, which is based on threshold 
and logical operation. We design the FE (feature enhancement) module on the skip-connection to emphasize the fine-grained 
features. In addition, our method varies the number of input channels in different branches at each stage of the network, so 
that the relationship between low and high frequency features can be learned. TBUnet yields 91.08 DSC on ISIC-2018 for 
melanoma segmentation, and achieves better performance than state-of-the-art medical image segmentation methods. Furthermore, experimental results with 82.48 DSC and 89.04 DSC obtained on the BUSI dataset and the Kvasir-SEG dataset 
show that TBUnet outperforms the advanced segmentation methods. Experiments demonstrate that TBUnet has excellent 
segmentation performance and generalisation capability.
</div>

## Citation
Please cite our paper if you find it useful: 
<pre>
@article{chen2023tbunet,
  title={TBUnet: A Pure Convolutional U-Net Capable of Multifaceted Feature Extraction for Medical Image Segmentation},
  author={Chen, LiFang and Li, Jiawei and Ge, Hongze},
  journal={Journal of Medical Systems},
  volume={47},
  number={1},
  pages={122},
  year={2023},
  publisher={Springer}
}
</pre>
