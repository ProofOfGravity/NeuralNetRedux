#include "OutputLayer.h"

OutputLayer::OutputLayer(std::string layer_name_in, std::string activation_type_in, std::string cost_type_in, int nNeurons_previous_layer_in, int nNeurons_this_layer_in, double learningRate_in):
	Layer(layer_name_in, activation_type_in, nNeurons_previous_layer_in, nNeurons_this_layer_in, learningRate_in),
	cost_type(cost_type_in)
{
}

void OutputLayer::PrintLayerInfo()
{

	for (int k = 0; k < nNeurons_previous_layer; k++)
	{
		std::cout << "   K";
	}

	std::cout << "\n";

	for (int j = 0; j < nNeurons_this_layer; j++)
	{

		std::cout << "J";

		for (int k = 0; k < nNeurons_previous_layer; k++)
		{

			std::cout << "  " << weight_matrix[(j * nNeurons_previous_layer) + k];

		}

		std::cout << "\n";
	}

	for (auto& i : neuron_vec)
	{
		std::cout << "\n";
		std::cout << "Bias " << i.Get_bias();
	}

	std::cout << std::endl;

	for (auto& i : neuron_vec)
	{
		std::cout << "\n";
		std::cout << "Probability = " << i.Get_activation();
	}

}

void OutputLayer::CalculateError(std::vector<double>& featureVecLabels)
{

	auto workingfunc = costFunctions.functions.find(cost_type);

	assert(workingfunc != costFunctions.functions.end());

	workingfunc->second(neuron_vec, featureVecLabels);

}
