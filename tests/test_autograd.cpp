#include "../src/core/autograd_graph.h"

#include <cassert>
#include <iostream>
#include <vector>

static Tensor	create_scalar(float value)
{
	Tensor	result(std::vector<size_t>(1, 1));

	result.data()[0] = value;
	return (result);
}

int	main()
{
	Tensor			a_value = create_scalar(2.0f);
	Tensor			b_value = create_scalar(3.0f);
	Variable		a(a_value, true);
	Variable		b(b_value, true);
	AutogradGraph	graph;
	Variable		*c;
	Variable		*d;

	c = graph.multiply(a, b);
	d = graph.add(*c, a);

	graph.backward(*d);

	assert(c->value().data()[0] == 6.0f);
	assert(d->value().data()[0] == 8.0f);

	assert(a.gradient().data()[0] == 4.0f);
	assert(b.gradient().data()[0] == 2.0f);

	std::cout << "Autograd graph tests passed"
		<< std::endl;

	return (0);
}
