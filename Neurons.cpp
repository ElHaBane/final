#include <iostream>
#include "network.h"

// DEFINITIONS FOR NEURON CLASS

neuron::neuron(double(*activation_function)(double x), const int BIAS, const float DEFAULT_ACTIVATED_VALUE) : // initializes activation_func at activation_function 
	activation_func(activation_function),
	bias(BIAS),
	activated_value(DEFAULT_ACTIVATED_VALUE),
	initial_activated_value(DEFAULT_ACTIVATED_VALUE),
	z(0)

{}

double neuron::get_activated_value() { return activated_value; };

void neuron::store_activated_value(double gross_value) {
	z = gross_value + bias;

	activated_value = activation_func(z);
	return;
}

double neuron::get_bias() { return bias; }
double neuron::get_z() { return z; }