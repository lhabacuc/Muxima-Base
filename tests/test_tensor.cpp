#include "../src/core/tensor.h"

#include <iostream>
#include <vector>

static int	test_shape()
{
	std::vector<size_t> shape;

	shape.push_back(2);
	shape.push_back(3);

	Tensor tensor(shape);

	if (tensor.shape().size() != 2)
		return (1);
	if (tensor.shape()[0] != 2)
		return (1);
	if (tensor.shape()[1] != 3)
		return (1);
	if (tensor.size() != 6)
		return (1);
	return (0);
}

static int	test_access()
{
	std::vector<size_t> shape;

	shape.push_back(2);
	shape.push_back(3);

	Tensor tensor(shape);

	tensor(0, 0) = 1.0f;
	tensor(0, 1) = 2.0f;
	tensor(0, 2) = 3.0f;
	tensor(1, 0) = 4.0f;
	tensor(1, 1) = 5.0f;
	tensor(1, 2) = 6.0f;

	if (tensor(0, 0) != 1.0f)
		return (1);
	if (tensor(1, 1) != 5.0f)
		return (1);
	if (tensor(1, 2) != 6.0f)
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
