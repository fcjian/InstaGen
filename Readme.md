<div align="center">
  
# InstaGen: Enhancing Object Detection by Training on Synthetic Dataset
[[Paper]()]
[[Project Page](https://fcjian.github.io/InstaGen)]
<be>
</div>

![method overview](resources/overview.png)

We are currently organizing the code for InstaGen to make it available as soon as possible.
If you are interested in our work, please star ⭐ our project. 
<br>

## Introduction

In this paper, we introduce a novel paradigm to enhance the ability of object detector, e.g., expanding categories or improving detection performance, by training on **synthetic dataset** generated from diffusion models. Specifically, we integrate an instance-level grounding head into a pre-trained, generative diffusion model, to augment it with the ability of localising arbitrary instances in the generated images. The grounding head is trained to align the text embedding of category names with the regional visual feature of the diffusion model, using supervision from an off-the-shelf object detector, and a novel self-training scheme on (novel) categories not covered by the detector. This enhanced version of diffusion model, termed as **InstaGen**, can serve as a data synthesizer for object detection. We conduct thorough experiments to show that, object detector can be enhanced while training on the synthetic dataset from InstaGen, demonstrating superior performance over existing state-of-the-art methods in open-vocabulary (+4.5 AP) and data-sparse (+1.2 ~ 5.2 AP) scenarios.

## Methodology
![method overview](resources/method.png)

## Synthetic Dataset
![method overview](resources/qualitative_result.png)

## Citation

If you find InstaGen useful in your research, please consider citing:

```
@inproceedings{feng2024instagen,
    title={InstaGen: Enhancing Object Detection by Training on Synthetic Dataset},
    author={Feng, Chengjian and Zhong, Yujie and Jie, Zequn and Xie, Weidi and Ma, Lin},
    journal={arxiv},
    year={2024}
}
```


