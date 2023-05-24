If run on colab than just run all cell if run on local machine first fullfil the requirements of builtin libaries tehn run all cell.

Q1
DoB - DD/MM/YY - 03/06/99
RollNo - M22CS061 - ABC - 061

Dataset:
	MTech: CIFAR10
Weight Initialization:
	MM - 06 Even Xavier
Data Augmentation Details:
	DD - 03 Odd 10 degree rotation and gaussian noise 
Pooling:
	MM - 06 Even AvgPool
Classification details:
	3+6+99 = 108 Even 0,2,4,6,8
Model Details:
	Feature Extraction: Layer 6 Conv and 1 pool with 12 filter
	FC Layer: 1FC with 512 nodes

Q2
	ABC - 061 odd
	Auto encoder with same above details
	Number of AE = 3
	Classification 1FC with 512 nodes



Question 1


Hyperparameter:
	Learning rate: 0.01
	Batch Size: 32
	Number of epoch: 10
	(other are mention early)

-Get only even class data
-Visulize Data
-Building model
-Xavier Initialization
-Loss and Accuracy Fuction
-Train and Test loop
-Loss and Accuracy plot vs Epoch


Question 2

Hyperparameters:
	Learning rate: 0.01
	Batch Size: 32
	Number of epoch: 10
	(other are mention early)


-Get only even class data
-Visulize Data
-Building an Auto encoder
-Xavier Initialization
-Train and Test Step
-loss and optimizer for encoder class
-FC1 Model
-train and test loop for FC
-Loss and Accuracy plot vs Epoch
