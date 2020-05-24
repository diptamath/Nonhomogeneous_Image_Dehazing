# Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing
The code for implementing the "Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing" (Accepted at NTIRE Workshop, CVPR 2020).

Preprint: https://arxiv.org/abs/2005.05999

## Highlights

- **Lightweight:** The proposed system is very lightweight as the total size of the model is around 21.7 MB.
- **Training Time Friendly:**  it is quite robust for different environments with various density of the haze or fog in the scene.
- **Fast:** it can process an HD image in 0.0145 seconds on average and can dehaze images from a video sequence in real-time.

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


