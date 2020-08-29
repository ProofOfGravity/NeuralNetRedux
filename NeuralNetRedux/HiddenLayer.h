#pragma once
#include "Layer.h"
class HiddenLayer :
    public Layer
{

public:

    HiddenLayer(std::string layer_name_in, std::string activation_type_in, int nNeurons_previous_layer_in, int nNeurons_this_layer_in, double learningRate_in = 10);

    virtual void CalculateError(std::vector<double>& featureVecLabels);

};

