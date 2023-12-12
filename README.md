# WorDepth: Variational Language Prior for Monocular Depth Estimation
>We propose a technique that improves monocular depth estimation models by incorporating human language guidance into training, since language can provide the depth model with geometry priors that are associated with semantics but not explicitly addressed in depth estimation datasets. To achieve this, we utilize CLIP, a visual-language model, to exploit the semantic priors learned from its large-scale training data. We use CLIP to encode language descriptions of the scene, which can be obtained through an image captioner, and employ a variational decoding framework to learn the distribution of the potential scenes corresponding to the language description. This variational approach creates a prior distribution on the scenes that can be effectively sampled by a conditional sampler that is later applied to regularize monocular depth model training. The method consistently improves depth estimation accuracy on the NYU Depth V2 and KITTI depth datasets. 


## Prepare
First install the environment by:
```
pip install -r requirements.txt
```

Then download the pretrained encoder backbone from [here](https://github.com/microsoft/Swin-Transformer), and then modify the pretrain path in the config files.

Then download the NYU Depth V2 Dataset from [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and then modify the dataset path in the config files.
## Training
There are 3 training stages in this code. Inside each stage run the model by:
```
cd Stage_1
sh train.sh
```

After successful training of Stage 1, then run Stage 2:
```
cd Stage_2
sh train.sh
```

After successful training of Stage 3, then run Stage 3:
```
cd Stage_3
sh train.sh
```
Then you should see the final results.



## Acknowledgements
Thanks to Ce Liu for opening source of the excellent work [VA-DepthNet](https://github.com/cnexah/VA-DepthNet/tree/main).
Thanks to Microsoft Research Asia for opening source of the excellent work [Swin Transformer](https://github.com/microsoft/Swin-Transformer).

