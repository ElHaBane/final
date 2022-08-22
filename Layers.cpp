#include <iostream>
#include <vector>
#include "network.h"


// DEFINITIONS FOR LAYER CLASS

// layer constructor
layer::layer(const int AMOUNT_OF_NEURONS, layer* the_next_layer, const int DEFAULT_BIAS, const float DEFAULT_WEIGHT_VALUE, const float DEFAULT_ACTIVATED_VALUE, double(*layers_activation_function)(double x)) {

	for (int i = 0; i < AMOUNT_OF_NEURONS; i++) { 
		layer_neurons.push_back(neuron(layers_activation_function, DEFAULT_BIAS, DEFAULT_ACTIVATED_VALUE));
	} // adds neurons to the layer

	next_layer = the_next_layer;
	neuron_amount = AMOUNT_OF_NEURONS;

	// sets the weights
	if (next_layer != NULL) { // if the value is not an output layer
		// instantiates weights for the layer's connection with the next layer
		for (int next_layers_i = 0; next_layers_i < (*next_layer).neuron_amount; next_layers_i++) { 
			weights.push_back(vector<double>(neuron_amount, DEFAULT_WEIGHT_VALUE)); // instantiates a row of weights that represent the layer's connection to one of the next layer's neurons
		}
	}
}



// passes the values of the layer into the neurons of the next layer
void layer::pass_forward() {

	for (int i_next_neuron = 0; i_next_neuron < (*next_layer).neuron_amount; i_next_neuron++) {

		neuron& next_neuron = (*next_layer).layer_neurons[i_next_neuron]; // refers to a neuron in the next layer

		double gross_value = 0; // the total inputted value of the neuron without the activation function and bias

		// sums the values the next_neuron recieves
		for (int i_current_neuron = 0; i_current_neuron < neuron_amount; i_current_neuron++) {
			gross_value += layer_neurons[i_current_neuron].get_activated_value() * weights[i_next_neuron][i_current_neuron] ;
		}
		
		next_neuron.store_activated_value(gross_value);

	}
	return;
}


// changes each of the layer's weights
void layer::adjust_weights(vector<vector<double>> const COST_CHANGES_PER_WEIGHTS) {
	int nxt_neur_difference = weights.size() - COST_CHANGES_PER_WEIGHTS.size(); // stores the differnece between the amount of neurons in the next layer and what the parameter believed the amount of neurons
	
	// loops through the neurons of the next layer
	for (int i_nxt_neuron = 0; i_nxt_neuron < weights.size() - nxt_neur_difference && i_nxt_neuron < COST_CHANGES_PER_WEIGHTS.size() + nxt_neur_difference; i_nxt_neuron++) {
		int crnt_neur_difference = weights[i_nxt_neuron].size() - COST_CHANGES_PER_WEIGHTS[i_nxt_neuron].size(); // stores the differnece between the amount of neurons in the current layer and what the parameter believed the amount of neurons
		
		// loops through the neurons of the current layer
		for (int i_crnt_neuron = 0; i_crnt_neuron < weights[i_nxt_neuron].size() - crnt_neur_difference && i_crnt_neuron < COST_CHANGES_PER_WEIGHTS[i_nxt_neuron].size() + crnt_neur_difference; i_crnt_neuron++) {
			weights[i_nxt_neuron][i_crnt_neuron] -= COST_CHANGES_PER_WEIGHTS[i_nxt_neuron][i_crnt_neuron];
		}
	}

	return;
}

// changes each of the layer's biases
void layer::adjust_biases(vector<double> const COST_CHANGES_PER_BIAS) {
	// loops through the neurons within the layer
	int neur_difference = layer_neurons.size() - COST_CHANGES_PER_BIAS.size(); // stores the differnece between the amount of neurons in the layer and what the parameter believed the amount of neurons
	for (int ineuon = 0; ineuon < layer_neurons.size() - neur_difference && ineuon < COST_CHANGES_PER_BIAS.size() + neur_difference; ineuon++) {
		cout << "change " << layer_neurons[ineuon].bias << endl;
		layer_neurons[ineuon].bias -= COST_CHANGES_PER_BIAS[ineuon];
		cout << "change " << layer_neurons[ineuon].bias << endl;

	}
	return;
}

int layer::get_num_of_weights() {
	return weights.size() * weights[0].size();
}

// sub class constructors
hiddenLayer::hiddenLayer(const int AMOUNT_OF_NEURONS, layer* the_next_layer, const int DEFAULT_BIAS, const int DEFAULT_WEIGHT_VALUE)
	: layer(AMOUNT_OF_NEURONS, the_next_layer, DEFAULT_WEIGHT_VALUE, DEFAULT_BIAS, 0) 
{}; // hidden layer constructor
inputLayer::inputLayer(const int AMOUNT_OF_NEURONS, layer* the_next_layer, const int DEFAULT_WEIGHT_VALUE, const int DEFAULT_ACTIVATED_VALUE) : 
	layer(AMOUNT_OF_NEURONS, the_next_layer, 0, DEFAULT_WEIGHT_VALUE, DEFAULT_ACTIVATED_VALUE, step)
{}; // input layer constructor
outputLayer::outputLayer(const int AMOUNT_OF_NEURONS, const int DEFAULT_BIAS) : 
	layer(AMOUNT_OF_NEURONS, NULL, DEFAULT_BIAS) 
{}; // output layer constructor
