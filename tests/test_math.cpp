#include "../src/core/math.h"

#include <iostream>
#include <vector>

static int	test_operations()
{
	std::vector<size_t> shape;
	Tensor	*a;
	Tensor	*b;
	Tensor	*result;

	shape.push_back(2);
	shape.push_back(3);

	a = new Tensor(shape);
	b = new Tensor(shape);

	a->data()[0] = 1.0f;
	a->data()[1] = 2.0f;
	a->data()[2] = 3.0f;
	a->data()[3] = 4.0f;
	a->data()[4] = 5.0f;
	a->data()[5] = 6.0f;

	b->data()[0] = 10.0f;
	b->data()[1] = 20.0f;
	b->data()[2] = 30.0f;
	b->data()[3] = 40.0f;
	b->data()[4] = 50.0f;
	b->data()[5] = 60.0f;

	result = new Tensor(add(*a, *b).shape());
	*result = add(*a, *b);

	if (result->data()[0] != 11.0f)
		return (1);
	if (result->data()[5] != 66.0f)
		return (1);

	*result = subtract(*b, *a);
	if (result->data()[0] != 9.0f)
		return (1);
	if (result->data()[5] != 54.0f)
		return (1);

	*result = multiply(*a, *b);
	if (result->data()[0] != 10.0f)
		return (1);
	if (result->data()[5] != 360.0f)
		return (1);

	*result = scalar_multiply(*a, 2.0f);
	if (result->data()[0] != 2.0f)
		return (1);
	if (result->data()[5] != 12.0f)
		return (1);

	delete a;
	delete b;
	delete result;
	return (0);
}

static int	test_matmul()
{
	std::vector<size_t> shape_a;
	std::vector<size_t> shape_b;
	Tensor	*a;
	Tensor	*b;

	shape_a.push_back(2);
	shape_a.push_back(3);

	shape_b.push_back(3);
	shape_b.push_back(2);

	a = new Tensor(shape_a);
	b = new Tensor(shape_b);

	a->data()[0] = 1.0f;
	a->data()[1] = 2.0f;
	a->data()[2] = 3.0f;
	a->data()[3] = 4.0f;
	a->data()[4] = 5.0f;
	a->data()[5] = 6.0f;

	b->data()[0] = 7.0f;
	b->data()[1] = 8.0f;
	b->data()[2] = 9.0f;
	b->data()[3] = 10.0f;
	b->data()[4] = 11.0f;
	b->data()[5] = 12.0f;

	{
		Tensor	result = matmul(*a, *b);

		if (result.shape()[0] != 2 || result.shape()[1] != 2)
			return (1);
		if (result.data()[0] != 58.0f)
			return (1);
		if (result.data()[1] != 64.0f)
			return (1);
		if (result.data()[2] != 139.0f)
			return (1);
		if (result.data()[3] != 154.0f)
			return (1);
	}

	delete a;
	delete b;
	return (0);
}

static int	test_transpose()
{
	std::vector<size_t> shape;
	Tensor	*tensor;

	shape.push_back(2);
	shape.push_back(3);

	tensor = new Tensor(shape);

	tensor->data()[0] = 1.0f;
	tensor->data()[1] = 2.0f;
	tensor->data()[2] = 3.0f;
	tensor->data()[3] = 4.0f;
	tensor->data()[4] = 5.0f;
	tensor->data()[5] = 6.0f;

	{
		Tensor	result = transpose(*tensor);

		if (result.shape()[0] != 3 || result.shape()[1] != 2)
			return (1);
		if (result.data()[0] != 1.0f)
			return (1);
		if (result.data()[1] != 4.0f)
			return (1);
		if (result.data()[2] != 2.0f)
			return (1);
		if (result.data()[3] != 5.0f)
			return (1);
		if (result.data()[4] != 3.0f)
			return (1);
		if (result.data()[5] != 6.0f)
			return (1);
	}

	delete tensor;
	return (0);
}

static int	test_add()
{
	std::vector<size_t> shape;
	Tensor	*a;
	Tensor	*b;
	Tensor	*result;

	shape.push_back(2);
	shape.push_back(3);

	a = new Tensor(shape);
	b = new Tensor(shape);

	a->data()[0] = 1.0f;
	a->data()[1] = 2.0f;
	a->data()[2] = 3.0f;
	a->data()[3] = 4.0f;
	a->data()[4] = 5.0f;
	a->data()[5] = 6.0f;

	b->data()[0] = 10.0f;
	b->data()[1] = 20.0f;
	b->data()[2] = 30.0f;
	b->data()[3] = 40.0f;
	b->data()[4] = 50.0f;
	b->data()[5] = 60.0f;

	result = new Tensor(add(*a, *b).shape());
	*result = add(*a, *b);

	if (result->data()[0] != 11.0f)
		return (1);
	if (result->data()[1] != 22.0f)
		return (1);
	if (result->data()[2] != 33.0f)
		return (1);
	if (result->data()[3] != 44.0f)
		return (1);
	if (result->data()[4] != 55.0f)
		return (1);
	if (result->data()[5] != 66.0f)
		return (1);

	delete a;
	delete b;
	delete result;
	return (0);
}

static int	test_softmax()
{
	std::vector<size_t> shape;
	Tensor	*tensor;

	shape.push_back(3);
	tensor = new Tensor(shape);

	tensor->data()[0] = 1.0f;
	tensor->data()[1] = 2.0f;
	tensor->data()[2] = 3.0f;

	{
		Tensor	result = softmax(*tensor);
		float	sum;

		sum = result.data()[0]
			+ result.data()[1]
			+ result.data()[2];

		if (sum < 0.999f || sum > 1.001f)
			return (1);
		if (result.data()[0] >= result.data()[1])
			return (1);
		if (result.data()[1] >= result.data()[2])
			return (1);
	}

	delete tensor;
	return (0);
}

static int	test_statistics()
{
	std::vector<size_t> shape;
	Tensor	*tensor;
	float	total;
	float	average;

	shape.push_back(4);
	tensor = new Tensor(shape);

	tensor->data()[0] = 1.0f;
	tensor->data()[1] = 2.0f;
	tensor->data()[2] = 3.0f;
	tensor->data()[3] = 4.0f;

	total = sum(*tensor);
	average = mean(*tensor);

	if (total != 10.0f)
		return (1);
	if (average != 2.5f)
		return (1);

	delete tensor;
	return (0);
}

static int	test_math_functions()
{
	std::vector<size_t> shape;
	Tensor	*tensor;

	shape.push_back(3);
	tensor = new Tensor(shape);

	tensor->data()[0] = 1.0f;
	tensor->data()[1] = 2.0f;
	tensor->data()[2] = 4.0f;

	{
		Tensor	result_exp = exp(*tensor);
		Tensor	result_log = log(*tensor);
		Tensor	result_sqrt = sqrt(*tensor);

		if (result_exp.data()[0] < 2.7f
			|| result_exp.data()[0] > 2.8f)
			return (1);
		if (result_log.data()[0] < -0.01f
			|| result_log.data()[0] > 0.01f)
			return (1);
		if (result_sqrt.data()[2] < 1.99f
			|| result_sqrt.data()[2] > 2.01f)
			return (1);
	}

	delete tensor;
	return (0);
}

static int	test_broadcast()
{
	std::vector<size_t> shape_a;
	std::vector<size_t> shape_b;
	Tensor	*a;
	Tensor	*b;

	shape_a.push_back(2);
	shape_a.push_back(3);

	shape_b.push_back(1);
	shape_b.push_back(3);

	a = new Tensor(shape_a);
	b = new Tensor(shape_b);

	a->data()[0] = 1.0f;
	a->data()[1] = 2.0f;
	a->data()[2] = 3.0f;
	a->data()[3] = 4.0f;
	a->data()[4] = 5.0f;
	a->data()[5] = 6.0f;

	b->data()[0] = 10.0f;
	b->data()[1] = 20.0f;
	b->data()[2] = 30.0f;

	{
		Tensor	result = add(*a, *b);

		if (result.data()[0] != 11.0f)
			return (1);
		if (result.data()[1] != 22.0f)
			return (1);
		if (result.data()[2] != 33.0f)
			return (1);
		if (result.data()[3] != 14.0f)
			return (1);
		if (result.data()[4] != 25.0f)
			return (1);
		if (result.data()[5] != 36.0f)
			return (1);
	}

	delete a;
	delete b;
	return (0);
}

static int	test_relu()
{
	std::vector<size_t> shape;
	Tensor	*tensor;

	shape.push_back(5);
	tensor = new Tensor(shape);

	tensor->data()[0] = -2.0f;
	tensor->data()[1] = -1.0f;
	tensor->data()[2] = 0.0f;
	tensor->data()[3] = 2.0f;
	tensor->data()[4] = 5.0f;

	{
		Tensor	result = relu(*tensor);

		if (result.data()[0] != 0.0f)
			return (1);
		if (result.data()[1] != 0.0f)
			return (1);
		if (result.data()[2] != 0.0f)
			return (1);
		if (result.data()[3] != 2.0f)
			return (1);
		if (result.data()[4] != 5.0f)
			return (1);
	}

	delete tensor;
	return (0);
}

static int	test_gelu()
{
	std::vector<size_t> shape;
	Tensor	*tensor;

	shape.push_back(3);
	tensor = new Tensor(shape);

	tensor->data()[0] = -1.0f;
	tensor->data()[1] = 0.0f;
	tensor->data()[2] = 1.0f;

	{
		Tensor	result = gelu(*tensor);

		if (result.data()[1] != 0.0f)
			return (1);
		if (result.data()[2] < 0.84f
			|| result.data()[2] > 0.85f)
			return (1);
	}

	delete tensor;
	return (0);
}

int	main()
{
	if (test_operations() != 0)
	{
		std::cout << "test_operations: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_operations: OK" << std::endl;

	if (test_matmul() != 0)
	{
		std::cout << "test_matmul: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_matmul: OK" << std::endl;
	
	if (test_transpose() != 0)
	{
		std::cout << "test_transpose: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_transpose: OK" << std::endl;
	
	if (test_add() != 0)
	{
		std::cout << "test_add: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_add: OK" << std::endl;
	
	if (test_softmax() != 0)
	{
		std::cout << "test_softmax: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_softmax: OK" << std::endl;

	if (test_statistics() != 0)
	{
		std::cout << "test_statistics: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_statistics: OK" << std::endl;

	if (test_math_functions() != 0)
	{
		std::cout << "test_math_functions: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_math_functions: OK" << std::endl;
	
	if (test_broadcast() != 0)
	{
		std::cout << "test_broadcast: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_broadcast: OK" << std::endl;

	if (test_relu() != 0)
	{
		std::cout << "test_relu: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_relu: OK" << std::endl;

	if (test_gelu() != 0)
	{
		std::cout << "test_gelu: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_gelu: OK" << std::endl;
	
	return (0);
}

