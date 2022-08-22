#include <iostream>
#include <string>
#include "network.h"
#include <iomanip>
#include "networks_expected_inputs_and_outputs.h" // has the expected inputs and outputs for the network

using namespace std;

void display_stage(int length, int columns, vector<float> values);

void print_spaces(int max_spaces);

void display_intro();

string get_string_input(const int MIN, const int MAX);

int get_integer_input(const int MIN, const int MAX);

int main() {

	neuralNetwork SquareBuilder(1, 16, 0.5);
	networkTrainer NetworkTrainer(SquareBuilder, EXPECTED_INPUTS, EXPECTED_OUTPUTS);

	cout << "what is your name?:" << endl;
	string name = get_string_input(0, 20);
	cout << "welcome " << name << endl;
	display_intro();

	while (true) {
		NetworkTrainer.store_network_weight_slopes();
	
		display_stage(20, 4, SquareBuilder.get_vector_output());
		cout << "how much time would you like to pass?" << endl;
		
		int time = get_integer_input(1, 50);

		for (int i = 0; i < time; i++) {
			NetworkTrainer.store_network_weight_slopes();
		}

	}
	
	return 0;
}

void display_intro() {
	cout << "Hello. This is a neural network. It is trying to create a square of ones. input the amount of generations you want to pass." << endl;
	return;
}



double step(double x) { return (x > 0) ? 1 : 0; } // One of the activation functions used by the neurons

double dtanh(double x) { return 1 / (pow(cosh(x), 2)); } // gets the derivative of tanh

// displays the numbers for the user in a four by four 
void display_stage(int length, int columns, vector<float> values) {
	
	
	for (int i = 0; i < columns; i++) { // prints four values per column
		cout << values[0 + (i * 4)];
		print_spaces(length / 4);
		cout << values[1 + (i * 4)];
		print_spaces(length / 4);
		cout << values[2 + (i * 4)];
		print_spaces(length / 4);
		cout << values[3 + (i * 4)];
		print_spaces(length / 4);

		
		cout << endl;
	}
}

void print_spaces(int max_spaces) {
	for (int space = 0; space < max_spaces; space++) { cout << " "; }
}

string get_string_input(const int MIN, const int MAX) {
	string input;

	cin >> setw(MAX) >> input;

	while (cin.good() == false) {// while input was not valid
		cout << "invalid input. Try again" << endl;

		// clears stream
		cin.clear();
		cin.ignore(INT_MAX, '\n');

		// retries retreiving input
		cin >> setw(MAX) >> input;
	}
	return input;
}

int get_integer_input(const int MIN, const int MAX) {
	int input = 0;

	cin >> setw(1) >> input;

	while (cin.good() == false || input < MIN || input > MAX) {// while input was not valid
		cout << "invalid input. Try again" << endl;

		// clears stream
		cin.clear();
		cin.ignore(INT_MAX, '\n');

		// retries retreiving input
		cin >> setw(1) >> input;
	}

	return input;
}