#include <iostream>
#include "ActivationFunctions.h"
#include "Layer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"
#include "InputLayer.h"
#include <memory>
#include "NeuralNet.h"


int main()
{
	std::vector<double> featureVec{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 1};
	
	std::vector<double> featureVecLabels{1, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	std::vector<double> featureVec2{ 0, 0, 1, 1, 0, 1, 1, 0, 0, 0 };

	std::vector<double> featureVecLabels2{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};

	std::vector<double> featureVec3{ 0, 1, 1, 1, 1, 1, 1, 1, 0, 0 };

	std::vector<double> featureVecLabels3{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

	NeuralNet nn;

	nn.CreateInputLayer(10);
	nn.CreateHiddenLayer("H1", "ReLU", 10, 0.1);
	nn.CreateHiddenLayer("H2", "tanh", 15, 0.1);
	nn.CreateOutputLayer("O1", "softmax", "softmaxCost", 10, 0.1);
	
	for (int i = 0; i < 100; i++)
	{
		nn.ForwardPass(featureVec);
		nn.BackPropAndUpdate(featureVecLabels);

		nn.ForwardPass(featureVec2);
		nn.BackPropAndUpdate(featureVecLabels2);

		nn.ForwardPass(featureVec3);
		nn.BackPropAndUpdate(featureVecLabels3);
	}


	nn.ForwardPass(featureVec);
	nn.PrintFinalGuess();

	std::cout << std::endl;

	nn.ForwardPass(featureVec2);
	nn.PrintFinalGuess();

	std::cout << std::endl;

	nn.ForwardPass(featureVec3);
	nn.PrintFinalGuess();
}

