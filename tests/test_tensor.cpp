


#include "../src/core/tensor.h"

#include <iostream>
#include <vector>
#include <stdexcept>

static int	test_shape()
{
	std::vector<size_t> shape;
	Tensor	*tensor;

	shape.push_back(2);
	shape.push_back(3);
	shape.push_back(4);

	tensor = new Tensor(shape);

	if (tensor->shape().size() != 3)
	{
		delete tensor;
		return (1);
	}
	if (tensor->shape()[0] != 2)
	{
		delete tensor;
		return (1);
	}
	if (tensor->shape()[1] != 3)
	{
		delete tensor;
		return (1);
	}
	if (tensor->shape()[2] != 4)
	{
		delete tensor;
		return (1);
	}
	if (tensor->size() != 24)
	{
		delete tensor;
		return (1);
	}

	delete tensor;
	return (0);
}

static int	test_access()
{
	std::vector<size_t> shape;
	std::vector<size_t> position;
	Tensor	*tensor;

	shape.push_back(2);
	shape.push_back(3);
	shape.push_back(4);

	tensor = new Tensor(shape);

	position.push_back(1);
	position.push_back(2);
	position.push_back(3);

	tensor->at(position) = 42.0f;

	if (tensor->at(position) != 42.0f)
	{
		delete tensor;
		return (1);
	}

	delete tensor;
	return (0);
}

static int	test_invalid_position()
{
	std::vector<size_t> shape;
	std::vector<size_t> position;
	Tensor	*tensor;

	shape.push_back(2);
	shape.push_back(3);
	shape.push_back(4);

	tensor = new Tensor(shape);

	position.push_back(2);
	position.push_back(0);
	position.push_back(0);

	try
	{
		tensor->at(position);
		delete tensor;
		return (1);
	}
	catch (const std::out_of_range&)
	{
		delete tensor;
		return (0);
	}
}

static int	test_invalid_dimension()
{
	std::vector<size_t> shape;
	Tensor	*tensor;

	shape.push_back(2);
	shape.push_back(0);
	shape.push_back(4);

	try
	{
		tensor = new Tensor(shape);
		delete tensor;
		return (1);
	}
	catch (const std::invalid_argument&)
	{
		return (0);
	}
}

static int	test_reshape()
{
	std::vector<size_t> shape;
	std::vector<size_t> new_shape;
	std::vector<size_t> position;
	Tensor	*tensor;

	shape.push_back(2);
	shape.push_back(3);
	shape.push_back(4);

	tensor = new Tensor(shape);

	position.push_back(1);
	position.push_back(2);
	position.push_back(3);

	tensor->at(position) = 42.0f;

	new_shape.push_back(4);
	new_shape.push_back(6);

	tensor->reshape(new_shape);

	if (tensor->shape().size() != 2)
	{
		delete tensor;
		return (1);
	}
	if (tensor->shape()[0] != 4)
	{
		delete tensor;
		return (1);
	}
	if (tensor->shape()[1] != 6)
	{
		delete tensor;
		return (1);
	}
	if (tensor->size() != 24)
	{
		delete tensor;
		return (1);
	}

	delete tensor;
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

	if (test_invalid_position() != 0)
	{
		std::cout << "test_invalid_position: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_invalid_position: OK" << std::endl;

	if (test_invalid_dimension() != 0)
	{
		std::cout << "test_invalid_dimension: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_invalid_dimension: OK" << std::endl;

	if (test_reshape() != 0)
	{
		std::cout << "test_reshape: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_reshape: OK" << std::endl;

	return (0);
}
