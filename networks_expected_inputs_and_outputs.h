#include <iostream>
#include <vector>

using namespace std;

// 0 = top left; 1 = top right; 2 = bottom left; 3 = bottom right; 4 = center

vector<vector<double>> const EXPECTED_INPUTS = {
	{1}
};

vector<vector<double>> const EXPECTED_OUTPUTS = {
	{0, 0, 0, 0
	,0, 1, 1, 0
	,0, 1, 1, 0
	,0, 0, 0, 0}
};