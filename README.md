# Nonhomogeneous_Image_Dehazing
Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing (Accepted at NTIRE Workshop, CVPR 2020)

Preprint: https://arxiv.org/abs/2005.05999

## Highlights

- **Simple:** Anchor-free, single-stage, light-head, no time-consuming post-processing. TTFNet only requires two detection heads for object localization and size regression, respectively.
- **Training Time Friendly:**  Our TTFNet outperforms a range of real-time detectors while suppressing them in training time. Moreover, super-fast TTFNet-18 and TTFNet-53 can reach 25.9 AP / 112 FPS only after 2 hours and 32.9 AP / 55 FPS after about 3 hours on the MS COCO dataset using 8 GTX 1080Ti.
- **Fast and Precise:** Our TTFNet-18/34/53 can achieve 28.1AP / 112FPS, 31.3AP / 87FPS, and 35.1AP / 54 FPS on 1 GTX 1080Ti.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system
```
python test

```

## Running the Training

Explain how to run the automated tests for this system
```
python train

```

# Quantitative Results
<img src="assets/cvpr_2.png" width="500"/>

# Qualitative Results
![](assets/cvpr_1.png)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Citation

Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@InProceedings{ Das_fast_deep_2020,
author = {Sourya Dipta Das and Saikat Dutta},
title = {Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
}

```


