#pragma once
#include <vector>
#include <assert.h>
#include <iostream>
#include <ctime>
#include <random>
#include "Neuron.h"
#include "ActivationFunctions.h"
#include "Random.h"

class Layer
{


public:

	Layer() = delete;
	Layer(std::string layer_name_in, std::string activation_type_in, int nNeurons_previous_layer_in, int nNeurons_this_layer_in, double learningRate_in = 10);


	std::vector<double>& GetWeightMatrix();
	std::vector<Neuron>& GetNeuronVector();

	void ForwardPass(std::vector<Neuron>& neurons_previous_layer);
	void BackPropagateError(std::vector<Neuron>& neurons_next_layer, std::vector<double>& weight_Matrix_next_layer);

	virtual void CalculateError(std::vector<double>& featureVecLabels);

	void UpdateWeightsandBiases(std::vector<Neuron>& neurons_previous_layer);

	virtual void PrintLayerInfo();

	int Get_nNeurons_this_layer();


	virtual void UpdateActivations(std::vector<double>& vec_in);

	
protected:

	std::string layer_name;
	std::string activation_type;

	int nNeurons_previous_layer;
	int nNeurons_this_layer;
	double learningRate;

	
	std::vector<Neuron> neuron_vec{};
	std::vector<double> weight_matrix{};

	ActivationFunctions activationfunctions;

	//Random is class for RNG that is a templete class 
	Random ran;
	double Init_Random_Number();

};

