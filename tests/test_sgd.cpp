#include "../src/core/variable.h"
#include "../src/training/sgd.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

static Tensor	create_tensor()
{
	std::vector<size_t> shape;

	shape.push_back(3);
	Tensor result(shape);
	result.data()[0] = 10.0f;
	result.data()[1] = 5.0f;
	result.data()[2] = -2.0f;
	return (result);
}

int	main()
{
	Tensor	value = create_tensor();
	Variable	variable(value, true);
	SGD			optimizer(0.1f);

	variable.gradient().data()[0] = 2.0f;
	variable.gradient().data()[1] = -4.0f;
	variable.gradient().data()[2] = 5.0f;

	optimizer.step(variable);

	assert(std::fabs(variable.value().data()[0] - 9.8f) < 0.0001f);
	assert(std::fabs(variable.value().data()[1] - 5.4f) < 0.0001f);
	assert(std::fabs(variable.value().data()[2] - (-2.5f)) < 0.0001f);

	optimizer.zero_grad(variable);

	assert(variable.gradient().data()[0] == 0.0f);
	assert(variable.gradient().data()[1] == 0.0f);
	assert(variable.gradient().data()[2] == 0.0f);

	std::cout << "SGD tests passed" << std::endl;
	return (0);
}
