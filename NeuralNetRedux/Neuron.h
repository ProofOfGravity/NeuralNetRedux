#pragma once
#include <vector>
#include <assert.h>
#include <iostream>

class Neuron
{


public:

	Neuron(double bias_in = 0, double activation_in = 0);

	//set_z will automatically take the bias into consideration, and add it to the passed in value
	void Set_z(double z_in);
	double Get_z();

	void Set_activation(double a_in);
	double Get_activation();

	void Set_bias(double bias_in);
	double Get_bias();

	void Set_error(double error_in);
	double Get_error();
	void Reset_error();

	void Set_derivativeResult(double derivative_in);
	double Get_derivativeResult();

private:

	double bias;
	double activation;
	double z;	

	double error;
	double derivativeResult;
	


};

