# Distracted_driver

Image recognition of whether a driver is distracted or not. This is from the Kaggle competition https://www.kaggle.com/c/state-farm-distracted-driver-detection. 10 classes are used and 4GB of data is given.

Two approaches have been attempted. One cold start CNN, which gave an OK performance, and one warm start using transfer learning from Resnet34. This worked far better.

## Getting Started

First, the data should be downloaded into the data folder found in the kaggle link about. After this is complete, running state_farm_main_fit.py using python3.6 will run a fastai model (using transfer learning from ReseNet34) to fit the data and create a predictions folder. 

The package requirements can be found in requirements.txt

## Authors

* **George Roberts**

## License

There is no license for this work.
