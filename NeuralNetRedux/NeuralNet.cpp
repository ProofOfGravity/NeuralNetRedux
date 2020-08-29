#include "NeuralNet.h"

NeuralNet::NeuralNet() :
	nNumber_of_layers(1)
{
}

void NeuralNet::CreateInputLayer(std::vector<double>& vec_in)
{
	LayerVec.push_back(std::make_unique<InputLayer>(InputLayer(vec_in)));
}

void NeuralNet::CreateInputLayer(int nNumber_of_neurons_in)
{
	LayerVec.push_back(std::make_unique<InputLayer>(InputLayer(nNumber_of_neurons_in)));
}

void NeuralNet::CreateHiddenLayer(std::string layer_name_in, std::string activation_type_in, int nNeurons_this_layer_in, double learningRate_in)
{
	int temp_nNeurons_previous = LayerVec.back()->Get_nNeurons_this_layer();
	LayerVec.push_back(std::make_unique<HiddenLayer>(HiddenLayer(layer_name_in, activation_type_in, temp_nNeurons_previous, nNeurons_this_layer_in, learningRate_in)));
	nNumber_of_layers++;
}

void NeuralNet::CreateOutputLayer(std::string layer_name_in, std::string activation_type_in, std::string cost_type_in, int nNeurons_this_layer_in, double learningRate_in)
{
	int temp_nNeurons_previous = LayerVec.back()->Get_nNeurons_this_layer();
	LayerVec.push_back(std::make_unique<OutputLayer>(OutputLayer(layer_name_in, activation_type_in, cost_type_in, temp_nNeurons_previous, nNeurons_this_layer_in, learningRate_in)));
	nNumber_of_layers++;
}

void NeuralNet::ForwardPass(std::vector<double>& featureVec_in)
{
	//updates 0 layer (AKA input layer) with new "activations", in other words, updates what will be eventually passed to next layer 
	UpDateInputs(featureVec_in);

	//iterate over every hidden layer and the output layer, thanks to different activation functions, it will also set the output for the final layer, defaulted to softmax
	for (int i = 1; i < nNumber_of_layers; i++)
	{
		LayerVec[i]->ForwardPass(LayerVec[i - 1]->GetNeuronVector());
	}

}

void NeuralNet::BackPropAndUpdate(std::vector<double>& featureVecLabels_in)
{
	//calculate the error in the last layer based on the input featureVecLabels; by default, this is a softmax layer
	LayerVec[nNumber_of_layers - 1]->CalculateError(featureVecLabels_in);

	//iterate over the remaining layers and backprop that error 
	for (int i = nNumber_of_layers - 2; i > 0; i--)
	{
		LayerVec[i]->BackPropagateError(LayerVec[i + 1]->GetNeuronVector(), LayerVec[i + 1]->GetWeightMatrix());
	}

	//now iterate over all layers and update the weights and biases (note, does not need to be called on the 0th layer, or the inputlayer, because there is nothing to update in that layer in terms of weights or biases)
	for (int i = nNumber_of_layers - 1; i > 0; i--)
	{
		LayerVec[i]->UpdateWeightsandBiases(LayerVec[i - 1]->GetNeuronVector());
	}
}

void NeuralNet::PrintFinalGuess()
{
	std::cout << std::endl;
	
	std::vector<Neuron>& tempVec = LayerVec[nNumber_of_layers - 1]->GetNeuronVector();

	for (auto& i : tempVec)
	{
		std::cout << i.Get_activation() << "\n";
	}

}

void NeuralNet::UpDateInputs(std::vector<double>& vec_in)
{
	LayerVec[0]->UpdateActivations(vec_in);
}
