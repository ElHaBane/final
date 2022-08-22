#include <iostream>
#include "network.h"

using namespace std;


networkTrainer::networkTrainer(neuralNetwork& the_neural_network, vector<vector<double>> expected_inputs, vector<vector<double>> expected_outputs) {
	neural_network = &the_neural_network;

	expected_input_output_values[iinput_expectation] = expected_inputs;
	expected_input_output_values[ioutput_expectation] = expected_outputs;
}

// applies the adjustments made to the weights of each layer
void networkTrainer::publish_networks_weight_slopes() {
	neuralNetwork& publisher_network = *neural_network;
	publisher_network.input_layer.adjust_weights(weight_slopes[0]); // publishes the input layer's weights
	for (int i = 0; i < publisher_network.hidden_layers.size(); i++) { publisher_network.hidden_layers[i].adjust_weights(weight_slopes[i + 1]); } // publishes hidden layer weights
	publisher_network.output_layer.adjust_weights(weight_slopes[weight_slopes.size() - 1]); // publishes output layer's weights

	return;
}

// gets the cost of the layer
double networkTrainer::get_layers_cost(hiddenLayer& const ACTUAL_LAYER, vector<double>& const LAYERS_EXPECTED_VALUES) {
	double cost = 0;
	
	for (int i = 0; i < ACTUAL_LAYER.layer_neurons.size(); i++) { 
		double neurons_value = ACTUAL_LAYER.layer_neurons[i].get_activated_value();
		cost += pow(neurons_value - LAYERS_EXPECTED_VALUES[i], 2);
	}
	return cost;
}
double networkTrainer::get_layers_cost(outputLayer& const ACTUAL_LAYER, vector<double>& const LAYERS_EXPECTED_VALUES) {
	double cost = 0;

	for (int i = 0; i < ACTUAL_LAYER.layer_neurons.size(); i++) {
		double neurons_value = ACTUAL_LAYER.layer_neurons[i].get_activated_value();
		cost += pow(neurons_value - LAYERS_EXPECTED_VALUES[i], 2);
	}
	return cost;
}

// gets the slope of the cost's change with respect to bias
double networkTrainer::get_cost_weight_slope(neuron& transmitter_neuron, neuron& primary_neuron, double expected_value) {
	return transmitter_neuron.get_activated_value() * dtanh(primary_neuron.get_z()) * 2 * (primary_neuron.get_activated_value() - expected_value);
}

// gets the slope of the cost's change with respect to weight
double networkTrainer::get_cost_bias_slope(neuron& primary_neuron, double expected_value) {
	return dtanh(primary_neuron.get_z()) * 2 * (primary_neuron.get_activated_value() - expected_value);
}

// gets the slope of the cost's change with respect to the activation function of a previous neuron
double networkTrainer::get_cost_prevav_slope(neuron& previous_neuron, neuron& primary_neuron, double expected_value) {
	return ((primary_neuron.get_z() - primary_neuron.get_bias()) / previous_neuron.get_activated_value()) * dtanh(primary_neuron.get_z()) * 2 * (primary_neuron.get_activated_value() - expected_value);
}

// gets the vector that represents the weight slope between two layers
vector<vector<double>> networkTrainer::get_layers_weight_slope(outputLayer primary_layer, hiddenLayer secondary_layer, vector<double>expected_primary_values) {
	vector<vector<double>> ws; // constains weight slopes

	for (int iprim = 0; iprim < primary_layer.layer_neurons.size(); iprim++) {
		vector<double> neurons_ws; // the weight slopes between the secondary layer and a primary neuron
		for (int isec = 0; isec < secondary_layer.layer_neurons.size(); isec++) {
			neurons_ws.push_back(get_cost_weight_slope(secondary_layer.layer_neurons[isec], primary_layer.layer_neurons[iprim], expected_primary_values[iprim]));
		}
		ws.push_back(neurons_ws); // adds the neuron's weight slopes to the layer's weight slope
	}
	return ws;
}
vector<vector<double>> networkTrainer::get_layers_weight_slope(hiddenLayer primary_layer, inputLayer secondary_layer, vector<double>expected_primary_values) {
	vector<vector<double>> ws; // constains weight slopes

	for (int iprim = 0; iprim < primary_layer.layer_neurons.size(); iprim++) {
		vector<double> neurons_ws; // the weight slopes between the secondary layer and a primary neuron
		for (int isec = 0; isec < secondary_layer.layer_neurons.size(); isec++) {
			neurons_ws.push_back(get_cost_weight_slope(secondary_layer.layer_neurons[isec], primary_layer.layer_neurons[iprim], expected_primary_values[iprim]));
		}
		ws.push_back(neurons_ws); // adds the neuron's weight slopes to the layer's weight slope
	}
	return ws;
}

vector<vector<double>> networkTrainer::get_layers_weight_slope(outputLayer primary_layer, inputLayer secondary_layer, vector<double>expected_primary_values) {
	vector<vector<double>> ws; // constains weight slopes

	for (int iprim = 0; iprim < primary_layer.layer_neurons.size(); iprim++) {
		vector<double> neurons_ws; // the weight slopes between the secondary layer and a primary neuron
		for (int isec = 0; isec < secondary_layer.layer_neurons.size(); isec++) {
			neurons_ws.push_back(get_cost_weight_slope(secondary_layer.layer_neurons[isec], primary_layer.layer_neurons[iprim], expected_primary_values[iprim]));
		}
		ws.push_back(neurons_ws); // adds the neuron's weight slopes to the layer's weight slope
	}
	return ws;
}

void networkTrainer:: store_network_weight_slopes() {
	weight_slopes.clear();
	neural_network->pass_completely();

	// loops through the expected outputs
	outputLayer& out = neural_network->output_layer;
	inputLayer& in = neural_network->input_layer;
	vector<double>& expected_out = expected_input_output_values[ioutput_expectation][0];

	// includes the weight slopes between the first and hidden layer
	weight_slopes.push_back(get_layers_weight_slope(out, in, expected_out));

	publish_networks_weight_slopes();
	weight_slopes.clear();
	return;
}