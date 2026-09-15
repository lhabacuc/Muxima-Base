#include "../src/core/variable.h"
#include "../src/training/optimizer.h"
#include "../src/training/sgd.h"
#include "../src/training/parameter_list.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

static Tensor	create_scalar(float value)
{
	std::vector<size_t> shape;

	shape.push_back(1);

	Tensor result(shape);
	result.data()[0] = value;

	return (result);
}

int	main()
{
	Tensor weight_value = create_scalar(10.0f);
	Tensor bias_value = create_scalar(5.0f);

	Variable weight(weight_value, true);
	Variable bias(bias_value, true);

	SGD optimizer(0.1f);
	ParameterList parameters;

	parameters.add(weight);
	parameters.add(bias);

	weight.gradient().data()[0] = 2.0f;
	bias.gradient().data()[0] = 3.0f;

	parameters.step(optimizer);

	assert(std::fabs(
		weight.value().data()[0] - 9.8f) < 0.0001f);

	assert(std::fabs(
		bias.value().data()[0] - 4.7f) < 0.0001f);

	parameters.zero_grad(optimizer);

	assert(weight.gradient().data()[0] == 0.0f);
	assert(bias.gradient().data()[0] == 0.0f);

	assert(parameters.size() == 2);

	std::cout << "Parameter list tests passed"
		<< std::endl;

	return (0);
}
