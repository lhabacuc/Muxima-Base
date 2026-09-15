#include "../src/core/tensor.h"

#include <iostream>
#include <vector>

static int	test_shape()
{
	std::vector<size_t> shape;

	shape.push_back(2);
	shape.push_back(3);
	shape.push_back(4);

	Tensor tensor(shape);

	if (tensor.shape().size() != 3)
		return (1);
	if (tensor.shape()[0] != 2)
		return (1);
	if (tensor.shape()[1] != 3)
		return (1);
	if (tensor.shape()[2] != 4)
		return (1);
	if (tensor.size() != 24)
		return (1);
	return (0);
}

static int	test_access()
{
	std::vector<size_t> shape;
	std::vector<size_t> position;

	shape.push_back(2);
	shape.push_back(3);
	shape.push_back(4);

	Tensor tensor(shape);

	position.push_back(1);
	position.push_back(2);
	position.push_back(3);

	tensor.at(position) = 42.0f;

	if (tensor.at(position) != 42.0f)
		return (1);
	return (0);
}

int	main()
{
	if (test_shape() != 0)
	{
		std::cout << "test_shape: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_shape: OK" << std::endl;

	if (test_access() != 0)
	{
		std::cout << "test_access: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_access: OK" << std::endl;

	return (0);
}
