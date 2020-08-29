#pragma once
#include "Neuron.h"
#include <cmath>
#include <functional>
#include <cassert>
#include <unordered_map>
#include <memory>
#include <string>

class CostFunctions
{

public:

	std::unordered_map<std::string, std::function<void(std::vector<Neuron>&, std::vector<double>&)>> functions;

	CostFunctions()
	{
		
		functions.emplace("softmaxCost", std::function<void(std::vector<Neuron>&, std::vector<double>&)>([](std::vector<Neuron>& neuron_in, std::vector<double>& feature_vector_labels)
			{

				assert(neuron_in.size() == feature_vector_labels.size());

				for (int i = 0; i < (int)neuron_in.size(); i++)
				{
					neuron_in[i].Set_error(neuron_in[i].Get_activation() - feature_vector_labels[i]);
				}

			}));

	}
	
};



