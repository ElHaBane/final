#include <iostream>
#include <vector>
#include "general_functions.h"

using namespace std;

class neuron {
private:
	double(*activation_func)(double x); // declares neuron's activation function
	double initial_activated_value;
	double activated_value; // declares the value of a neuron's value after the activation function is used
	double z; // declares the value of the neuron's value before it was activated
	int bias; // declares an inputted value's tendency to be either negative or positive

public:
	neuron(double(*activation_function)(double x), const int BIAS, const float DEFAULT_ACTIVATED_VALUE = 0); // declares constructor 

	double get_activated_value(); // gets the activated value

	void store_activated_value(double gross_value); // stores the activated value

	double get_bias();
	double get_z();

	friend class layer;
	};

// layers --------------------------------------------------------------------
class layer {
private:
	int neuron_amount;
	vector<vector<double>> weights; // declares the weights for the connections


public:
	layer(const int AMOUNT_OF_NEURONS, layer* the_next_layer = NULL, const int DEFAULT_BIAS = 1, const float DEFAULT_WEIGHT_VALUE = 1, const float DEFAULT_ACTIVATED_VALUE = 0, double(*layers_activation_function)(double x) = tanh); // declares constructor
	layer* next_layer; // points to the layer that's in front of this layer

	vector<neuron> layer_neurons; // declares a vector that holds the layer's neurons

	void pass_forward(); // passes the values of the layer's neurons to the next layer's neurons
	void adjust_weights(vector<vector<double>> const COST_CHANGES_PER_WEIGHTS);
	void adjust_biases(vector<double> const COST_CHANGES_PER_BIAS);
	int get_num_of_weights();
};


// layer sub classes----------------------------------------------------------
// initializing constructors and changing variables
class hiddenLayer : public layer { public: hiddenLayer(const int AMOUNT_OF_NEURONS = 1, layer* the_next_layer = NULL, const int DEFAULT_BIAS = 1, const int DEFAULT_WEIGHT_VALUE = 1); }; // declares constructor
class inputLayer : public layer { public: inputLayer(const int AMOUNT_OF_NEURON = 1, layer* the_next_layer = NULL, const int DEFAULT_WEIGHT_VALUE = 1, const int DEFAULT_ACTIVATED_VALUE = 0); }; // declares constructor
class outputLayer : public layer { public: outputLayer(const int AMOUNT_OF_NEURONS = 1, const int DEFAULT_BIAS = 1); }; // declares constructor


// neural network class ---------------------------------------------------------
class neuralNetwork {
private:
	inputLayer input_layer;
	vector<hiddenLayer> hidden_layers; // stores the hidden layer
	outputLayer output_layer; // stores the output layer

public:
	neuralNetwork(const int NUM_OF_INPUTS, const int NUM_OF_OUTPUTS, const double DEFAULT_INPUT_VALUE = 0, const int NUM_OF_HIDDEN_LAYERS = 0, const int NUM_OF_HIDDEN = 2); // declares constructor

	const int NUMBER_OF_LAYERS = hidden_layers.size() + 2; // stores the number of layers the network has
	vector<float> get_vector_output(); // gets the output as a vector
	void view_layer_values(const int LAYER_INDEX); // displays the activated values of a layer's neurons
	void pass_completely(); // passes the neural network's activated values from the input to the input layer
	
	friend class networkTrainer;
};


// network trainer----------------------------------------------------------------
class networkTrainer {
private:
	enum input_output {iinput_expectation, ioutput_expectation}; // stores the values for the input and output expectation indexes
	vector<vector<double>> expected_input_output_values[2]; // declares the two vectors that store the expected outputs for each layer within the network
	/*
	expected_values_per_layer =
			{1, 2, 3}				{4, 5, 6}
	input:	{1, 2, 3}	,	output:	{4, 5, 6}
			{1, 2, 3}				{4, 5, 6}
	*/
	vector<vector<vector<double>>> weight_slopes; // declares the slopes the network trainer gathers from each layer of the neural network
	vector<vector<double>> bias_slopes; // declares the slopes the network trainer gathers from each layer of the neural network
	
	void publish_networks_weight_slopes();// applies the weight slopes to the network
	void publish_networks_bias_slopes(); // applies the bias slopes to the network

public:
	
	networkTrainer(neuralNetwork& the_neural_network, vector<vector<double>> expected_inputs, vector<vector<double>> expected_outputs);
	neuralNetwork* neural_network;

	double get_layers_cost(hiddenLayer& const ACTUAL_LAYER, vector<double>& const LAYERS_EXPECTED_VALUES);
	double get_layers_cost(outputLayer& const ACTUAL_LAYER, vector<double>& const LAYERS_EXPECTED_VALUES);

	double get_cost_weight_slope(neuron& transmitter_neuron, neuron& reciever_neuron, double expected_value);
	double get_cost_bias_slope(neuron& n, double expected_value);
	double get_cost_prevav_slope(neuron& previous_neuron, neuron& current_neuron, double expected_value);

	void store_network_weight_slopes(); // stores the weights of the network

	vector<vector<double>> get_layers_weight_slope(outputLayer primary_layer, hiddenLayer secondary_layer, vector<double>expected_primary_values); // gets the weight slopes of two layers
	vector<vector<double>> get_layers_weight_slope(hiddenLayer primary_layer, inputLayer secondary_layer, vector<double>expected_primary_values); // gets the weight slopes of two layers
	vector<vector<double>> get_layers_weight_slope(outputLayer primary_layer, inputLayer secondary_layer, vector<double>expected_primary_values);
};