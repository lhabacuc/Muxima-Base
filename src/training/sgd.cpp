#include "sgd.h"

#include <stdexcept>

SGD::SGD(float learning_rate)
	: _learning_rate(learning_rate)
{
	if (learning_rate <= 0.0f)
		throw std::invalid_argument("Learning rate must be positive");
}

void	SGD::step(Variable& variable)
{
	size_t	i;

	if (!variable.requires_grad())
		return ;
	i = 0;
	while (i < variable.value().size())
	{
		variable.value().data()[i] -=
			_learning_rate * variable.gradient().data()[i];
		i++;
	}
}

void	SGD::zero_grad(Variable& variable)
{
	variable.zero_grad();
}

float	SGD::learning_rate() const
{
	return (_learning_rate);
}
