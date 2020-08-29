#include "Layer.h"

Layer::Layer(std::string layer_name_in, std::string activation_type_in, int nNeurons_previous_layer_in, int nNeurons_this_layer_in, double learningRate_in) :
	layer_name(layer_name_in),
	activation_type(activation_type_in),
	nNeurons_previous_layer(nNeurons_previous_layer_in),
	nNeurons_this_layer(nNeurons_this_layer_in),
	learningRate(learningRate_in)
{
	for (int i = 0; i < (nNeurons_previous_layer * nNeurons_this_layer); i++)
	{
		weight_matrix.push_back(Init_Random_Number());
	}

	for (int i = 0; i < nNeurons_this_layer; i++)
	{
		neuron_vec.push_back(Neuron(Init_Random_Number()));
	}

}

std::vector<double>& Layer::GetWeightMatrix()
{
	return weight_matrix;
}

std::vector<Neuron>& Layer::GetNeuronVector()
{
	return neuron_vec;
}

void Layer::ForwardPass(std::vector<Neuron>& neurons_previous_layer)
{

	for (int j = 0; j < nNeurons_this_layer; j++)
	{

		//sum weights and biases for a given input layer neurons and set Z
		double temp = 0;

		for (int k = 0; k < nNeurons_previous_layer; k++)
		{

			temp += neurons_previous_layer[k].Get_activation() * weight_matrix[(j * nNeurons_previous_layer) + k];

		}
		
		neuron_vec[j].Set_z(temp);

	}

	//Now that Z has been set, check to make sure the activation function is real and then calculate all a's for this layers neuron vec
	auto workingfunc = activationfunctions.functions.find(activation_type);

	assert(workingfunc != activationfunctions.functions.end());

	workingfunc->second(neuron_vec);

}

void Layer::BackPropagateError(std::vector<Neuron>& neurons_next_layer, std::vector<double>& weight_Matrix_next_layer)
{

	auto workingfunc = activationfunctions.functions.find(activation_type + "Derivative");

	assert(workingfunc != activationfunctions.functions.end());

	workingfunc->second(neuron_vec);

	for (int k = 0;  k < nNeurons_this_layer; k++ )
	{

		double temp = 0;

		for (int j = 0; j < static_cast<int>(neurons_next_layer.size()); j++)
		{
			temp += neurons_next_layer[j].Get_error() * weight_Matrix_next_layer[(j * nNeurons_this_layer) + k];
		}

		neuron_vec[k].Set_error(temp * neuron_vec[k].Get_derivativeResult());

	}

}

void Layer::CalculateError(std::vector<double>& featureVecLabels)
{
}

void Layer::UpdateWeightsandBiases(std::vector<Neuron>& neurons_previous_layer)
{

	for (int j = 0; j < nNeurons_this_layer; j++)
	{

		for (int k = 0; k < nNeurons_previous_layer; k++)
		{

			weight_matrix[(j * nNeurons_previous_layer) + k] -= (neurons_previous_layer[k].Get_activation() * neuron_vec[j].Get_error()) * learningRate;

		}

	}
	
	for (auto& i : neuron_vec)
	{
		i.Set_bias(i.Get_bias() - (i.Get_error() * learningRate));
	}

}


void Layer::PrintLayerInfo()
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

}

int Layer::Get_nNeurons_this_layer()
{
	return nNeurons_this_layer;
}

void Layer::UpdateActivations(std::vector<double>& vec_in)
{
}

double Layer::Init_Random_Number()
{
	return ran.random<double>(-1.0, 1.0);
}
