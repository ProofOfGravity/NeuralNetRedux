#include "Neuron.h"

Neuron::Neuron(double bias_in, double activation_in) :
	bias(bias_in),
	activation(activation_in),
	z(0),
	error(0),
	derivativeResult(0)
{
}


void Neuron::Set_z(double z_in)
{
	z = z_in + bias;
}

double Neuron::Get_z()
{
	return z;
}

void Neuron::Set_activation(double a_in)
{
	activation = a_in;
}

double Neuron::Get_activation()
{
	return activation;
}

void Neuron::Set_bias(double bias_in)
{
	bias = bias_in;
}

double Neuron::Get_bias()
{
	return bias;
}

void Neuron::Set_error(double error_in)
{
	error = error_in;
}

double Neuron::Get_error()
{
	return error;
}

void Neuron::Reset_error()
{
	error = 0;
}

void Neuron::Set_derivativeResult(double derivative_in)
{
	derivativeResult = derivative_in;
}

double Neuron::Get_derivativeResult()
{
	return derivativeResult;
}
