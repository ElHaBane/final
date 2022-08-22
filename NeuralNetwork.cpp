#include <iostream>
#include "network.h"

// DEFINITIONS FOR NEURAL NETWORK CLASS
neuralNetwork::neuralNetwork(const int NUM_OF_INPUTS, const int NUM_OF_OUTPUTS, const double DEFAULT_INPUT_VALUE, const int NUM_OF_HIDDEN_LAYERS, const int NUM_OF_HIDDEN) {
	
	// initializes output layer
	output_layer = outputLayer(NUM_OF_OUTPUTS);
	

	// initializes hidden layers
	if (NUM_OF_HIDDEN_LAYERS > 0) {
		hidden_layers = vector<hiddenLayer>(NUM_OF_HIDDEN_LAYERS); // creates enough space for the hidden layers

		hidden_layers.back() = hiddenLayer(NUM_OF_HIDDEN, &output_layer);
		for (int hid_lay_i = NUM_OF_HIDDEN_LAYERS - 2; hid_lay_i >= 0; hid_lay_i--) { // connects the hidden layers to each other
			hidden_layers[hid_lay_i] = hiddenLayer(NUM_OF_HIDDEN, &hidden_layers[hid_lay_i + 1]);
		}
	}

	
	// initializes input layer
	if (NUM_OF_HIDDEN_LAYERS > 0) {input_layer = inputLayer(NUM_OF_INPUTS, &hidden_layers[0], 1, DEFAULT_INPUT_VALUE); } // if there is hidden layers
	else {input_layer = inputLayer(NUM_OF_INPUTS, &output_layer, 1, DEFAULT_INPUT_VALUE);} // if there isn't hidden layers
}

// displays a layer within neuralNetwork
void neuralNetwork::view_layer_values(const int LAYER_INDEX) {
	// if the selected index is the input layer
	if (LAYER_INDEX == 0) {
		for (neuron& n : input_layer.layer_neurons) {cout << n.get_activated_value() << endl;} // dispays all the input layer's activated values
	}

	// if the selected index is a hidden layer
	else if (LAYER_INDEX > 0 && LAYER_INDEX <= hidden_layers.size()) {
		for (neuron& n : hidden_layers[LAYER_INDEX - 1].layer_neurons) {cout << n.get_activated_value() << endl;} // displays all the activated values of the selected layer
	}	

	// if the selected index is not an input or hidden layer
	else {
		for (neuron& n : output_layer.layer_neurons) { cout << n.get_activated_value() << endl; } // dispays all the output layer's activated values
	}

	return;
}

void neuralNetwork::pass_completely() {
	input_layer.pass_forward(); // passes input layer's values
	for (hiddenLayer layer : hidden_layers) { layer.pass_forward(); } // passes the ouput layers' values
	return;
}

vector<float> neuralNetwork::get_vector_output() {
	vector<float>vector_out;

	for (int i = 0; i < output_layer.layer_neurons.size(); i++) { // adds the values to the vector
		vector_out.push_back(output_layer.layer_neurons[i].get_activated_value()); 
	}

	return vector_out;
}