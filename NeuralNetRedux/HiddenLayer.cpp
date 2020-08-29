#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(std::string layer_name_in, std::string activation_type_in, int nNeurons_previous_layer_in, int nNeurons_this_layer_in, double learningRate_in) :
	Layer(layer_name_in, activation_type_in, nNeurons_previous_layer_in, nNeurons_this_layer_in, learningRate_in)
{
}

void HiddenLayer::CalculateError(std::vector<double>& featureVecLabels)
{
}
