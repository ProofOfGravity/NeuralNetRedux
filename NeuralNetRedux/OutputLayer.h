#pragma once
#include "Layer.h"
#include "CostFunctions.h"
class OutputLayer :
    public Layer
{


public:

    OutputLayer(std::string layer_name_in, std::string activation_type_in, std::string cost_type_in, int nNeurons_previous_layer_in, int nNeurons_this_layer_in, double learningRate_in = 0.01);


     virtual void PrintLayerInfo() override;


     virtual void CalculateError(std::vector<double>& featureVecLabels) override;



private:

    std::string cost_type;

    CostFunctions costFunctions;

};

