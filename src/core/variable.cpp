#include "variable.h"

Variable::Variable(
	const Tensor& value,
	bool requires_grad)
	: _value(value),
	_gradient(value.shape()),
	_requires_grad(requires_grad)
{
}

Tensor&	Variable::value()
{
	return (_value);
}

Tensor&	Variable::gradient()
{
	return (_gradient);
}

const Tensor&	Variable::value() const
{
	return (_value);
}

const Tensor&	Variable::gradient() const
{
	return (_gradient);
}

bool	Variable::requires_grad() const
{
	return (_requires_grad);
}

void	Variable::zero_grad()
{
	size_t	i;

	i = 0;
	while (i < _gradient.size())
	{
		_gradient.data()[i] = 0.0f;
		i++;
	}
}
