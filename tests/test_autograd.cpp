#include "../src/core/autograd.h"

#include <cassert>
#include <iostream>
#include <vector>

int	main()
{
	std::vector<size_t>	shape;
	Tensor				a_value(std::vector<size_t>(1, 2));
	Tensor				b_value(std::vector<size_t>(1, 2));
	Tensor				gradient(std::vector<size_t>(1, 2));
	Variable			a(a_value, true);
	Variable			b(b_value, true);

	a.value().data()[0] = 2.0f;
	a.value().data()[1] = 3.0f;

	b.value().data()[0] = 4.0f;
	b.value().data()[1] = 5.0f;

	gradient.data()[0] = 1.0f;
	gradient.data()[1] = 1.0f;

	Autograd::multiply_backward(a, b, gradient);

	assert(a.gradient().data()[0] == 4.0f);
	assert(a.gradient().data()[1] == 5.0f);

	assert(b.gradient().data()[0] == 2.0f);
	assert(b.gradient().data()[1] == 3.0f);

	std::cout << "Autograd tests passed" << std::endl;
	return (0);
}
