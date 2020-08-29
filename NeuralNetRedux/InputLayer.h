#pragma once
#include "Layer.h"
class InputLayer :
    public Layer
{

public:

    InputLayer(std::vector<double> vec_in);
    InputLayer(int nnNeurons_this_layer_in);

    virtual void UpdateActivations(std::vector<double>& vec_in) override;

    virtual void PrintLayerInfo() override;

};

