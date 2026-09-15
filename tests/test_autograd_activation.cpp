#include "../src/core/autograd_graph.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int	main()
{
	Tensor			value(
		std::vector<size_t>(1, 3));
	Variable		input(value, true);
	AutogradGraph	graph;
	Variable		*output;

	input.value().data()[0] = -2.0f;
	input.value().data()[1] = 0.0f;
	input.value().data()[2] = 3.0f;

	output = graph.relu(input);

	graph.backward(*output);

	assert(output->value().data()[0] == 0.0f);
	assert(output->value().data()[1] == 0.0f);
	assert(output->value().data()[2] == 3.0f);

	assert(input.gradient().data()[0] == 0.0f);
	assert(input.gradient().data()[1] == 0.0f);
	assert(input.gradient().data()[2] == 1.0f);

	std::cout << "Autograd activation tests passed"
		<< std::endl;

	return (0);
}
