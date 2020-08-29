#pragma once
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"

class NeuralNet
{

public:

	NeuralNet();

	//layers MUST be created in order in this current version 
	void CreateInputLayer(std::vector<double>& vec_in);
	void CreateInputLayer(int nNumber_of_neurons_in);
	void CreateHiddenLayer(std::string layer_name_in, std::string activation_type_in, int nNeurons_this_layer_in, double learningRate_in);
	void CreateOutputLayer(std::string layer_name_in, std::string activation_type_in, std::string cost_type_in, int nNeurons_this_layer_in, double learningRate_in);

	void ForwardPass(std::vector<double>& featureVec_in);
	void BackPropAndUpdate(std::vector<double>& featureVecLabels_in);

	void StochasticGradientBatch(std::vector<std::vector<double>>& featureVec_in, std::vector<std::vector<double>>& featureVecLabels_in);

	void PrintFinalGuess();

private:

	void UpDateInputs(std::vector<double>& vec_in);

private:

	int nNumber_of_layers;
	std::vector<std::unique_ptr<Layer>> LayerVec;

};

