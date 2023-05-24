If run on colab than just run all cell if run on local machine first fullfil the requirements of builtin libaries tehn run all cell.

Question 1


Hyperparameter:

	Number of epochs: 5
	Batch size: 32
	Learning Rate: 0.2
	Hidden layer: 10


-Importing req libraries and load dataset and datloader

-Visulize data

-building model

-sigmoid as activation function

	-Loss and Accuracy Function

	-Train and Test Loop

	-Build model with 1 hidden layer
	-Build model with 2 hidden layer
	-Build model with 3 hidden layer
	-Build model with 4 hidden layer

-Building model as Tanh an Activation function

	-Build Model with different hidden layer

-Methods to overcome the vanishing gradient problem
	-use ReLu as activation function
	-Hyper parameter Turning


Question 2

-Import files

-Download custom data

-Visualize image

-Transforming data

-Load image data using ImageFolder

-DataLoader

-Create Dropout Class

-Create Model

-Dropout Function

-Train and Test Loop (Function)

-Validate function

-Train model without any Regularization
	Hyperparameters:
		Hidden units: 13
		Hidden Layers: 3
		Learning Rate: 0.003
		Epochs: 25
		Batch size: 32

	Result After 25 epochs:
		Train Loss: 1.0215 Train Accuracy: 0.5371
		Test Loss: 1.1277 Test Accuracy: 0.1837 

-Train model with L1 Regularization

	L1 regu = loss + lambda * weights norm
	
	Hyperparameters:
		Hidden units: 13
		Hidden Layers: 3
		Learning Rate: 0.003
		Epochs: 25
		Batch size: 32
		Regularization lambda: 0.0018 (not too small and to big)

	Result After 25 epochs:
		Train Loss: 1.4341 Train Accuracy: 0.6543
		Test Loss: 1.1389 Test Accuracy: 0.2012

-Train Model with L2 Regularization

	l2 regu = loss + lambda * weights square norm	
	
	Hyperparameters:
		Hidden units: 13
		Hidden Layers: 3
		Learning Rate: 0.0033
		Epochs: 25
		Batch size: 32
		Regularization lambda: 0.0019 (not too small and to big)

	Result After 25 epochs:
		Train Loss: 1.0614 Train Accuracy: 0.8176
		Test Loss: 1.0029 Test Accuracy: 0.2077


-Train model with Dropout
	
	Drop ramdomly some nodes from each hidden layer

-checking gradient

	https://miro.medium.com/max/640/0*zCHCJ4EOHMgWmIbJ.webp
