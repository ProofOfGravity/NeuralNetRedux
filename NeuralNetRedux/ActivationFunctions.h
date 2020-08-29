#pragma once
#include "Neuron.h"
#include <cmath>
#include <functional>
#include <cassert>
#include <unordered_map>
#include <memory>
#include <string>


class ActivationFunctions
{

public:

	ActivationFunctions() 
	{

		//places two lambdas in the unordered map, both sigmoid and sigmoid derivative 
		functions.emplace("sigmoid", std::function<void(std::vector<Neuron>&)>([](std::vector<Neuron>& neuron_in) {   for (auto& i : neuron_in) { i.Set_activation((1 / (1 + std::exp(-i.Get_z())))); }  }));

		functions.emplace("sigmoidDerivative", std::function<void(std::vector<Neuron>&)>([](std::vector<Neuron>& neuron_in) 
			{   

				for (auto& i : neuron_in) 
				{ 
					double temp = (1 / (1 + std::exp(-i.Get_z())));
					i.Set_derivativeResult(temp * (1-temp)); 
				}  

			}));


		//places two lambaas in the unordered map, both tanh and tanh derivative

		functions.emplace("tanh", std::function<void(std::vector<Neuron>&)>([](std::vector<Neuron>& neuron_in)
			{

				for (auto& i : neuron_in)
				{
					double z = i.Get_z();
					double ePos = std::exp(z);
					double eNeg = std::exp(-z);
					i.Set_activation((ePos - eNeg) / (ePos + eNeg));
				}

			}));

		functions.emplace("tanhDerivative", std::function<void(std::vector<Neuron>&)>([](std::vector<Neuron>& neuron_in)
			{

				for (auto& i : neuron_in)
				{
					i.Set_derivativeResult(1 - (i.Get_activation() * i.Get_activation()));
				}

			}));


		//places functions for softmax into unordered map 

		functions.emplace("softmax", std::function<void(std::vector<Neuron>&)>([](std::vector<Neuron>& neuron_in)
			{

				double softmaxDenominator = 0;
				for (auto& i : neuron_in)
				{
					softmaxDenominator += std::exp(i.Get_z());
				}

				for (auto& i : neuron_in)
				{
					i.Set_activation(std::exp(i.Get_z()) / softmaxDenominator);
				}

			}));



		//functions for ReLU

		functions.emplace("ReLU", std::function<void(std::vector<Neuron>&)>([](std::vector<Neuron>& neuron_in)
			{
				
				for (auto& i : neuron_in)
				{
					i.Set_activation(std::max(0.0, i.Get_z()));
				}

			}));


		functions.emplace("ReLUDerivative", std::function<void(std::vector<Neuron>&)>([](std::vector<Neuron>& neuron_in)
			{

				for (auto& i : neuron_in)
				{
					if (i.Get_z() < 0)
					{
						i.Set_derivativeResult(0);
					}
					else
					{
						i.Set_derivativeResult(1);
					}
				}

			}));

		//functions for Leaky ReLU

		functions.emplace("LeakyReLU", std::function<void(std::vector<Neuron>&)>([](std::vector<Neuron>& neuron_in)
			{

				for (auto& i : neuron_in)
				{
					if (i.Get_z() < 0)
					{
						i.Set_activation(i.Get_z() * 0.1);
					}
					else
					{
						i.Set_activation(i.Get_z());
					}
				}

			}));


		functions.emplace("LeakyReLUDerivative", std::function<void(std::vector<Neuron>&)>([](std::vector<Neuron>& neuron_in)
			{

				for (auto& i : neuron_in)
				{
					if (i.Get_z() < 0)
					{
						i.Set_derivativeResult(0.1);
					}
					else
					{
						i.Set_derivativeResult(1);
					}
				}

			}));
	}

	std::unordered_map<std::string, std::function<void(std::vector<Neuron>&)>> functions;

};

