# Distracted_driver

Image recognition of whether a driver is distracted or not. This is from the Kaggle competition https://www.kaggle.com/c/state-farm-distracted-driver-detection. 10 classes are used and 4GB of data is given.

Two approaches have been attempted. One cold start CNN, which gave an OK performance, and one warm start using transfer learning from Resnet34. This worked far better.

## Getting Started

All functions and classes are currently contained within main.py. By default, running the file will use fastAI and pretrained weights, although the class Explore_Data can be used for some basic EDA.

## Authors

* **George Roberts**

## License

There is no license for this work.
