#include "InputLayer.h"

InputLayer::InputLayer(std::vector<double> vec_in):
	Layer("InputLayer", "sigmoid", 0, 0, 0)
{
	nNeurons_this_layer = static_cast<int>(vec_in.size());

	for (int i = 0; i < nNeurons_this_layer; i++)
	{
		neuron_vec.push_back(Neuron(0, vec_in[i]));
	}
}

InputLayer::InputLayer(int nnNeurons_this_layer_in) :
	Layer("InputLayer", "sigmoid", 0, nnNeurons_this_layer_in, 0)
{
}

void InputLayer::UpdateActivations(std::vector<double>& vec_in)
{
	assert(nNeurons_this_layer = static_cast<int>(vec_in.size()));

	for (int i = 0; i < nNeurons_this_layer; i++)
	{
		neuron_vec[i].Set_activation(vec_in[i]);
	}
}

void InputLayer::PrintLayerInfo()
{

	std::cout << std::endl;

	for (int i = 0; i < nNeurons_this_layer; i++)
	{
		std::cout << neuron_vec[i].Get_activation() << "\n";
	}

}
