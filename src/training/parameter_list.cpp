#include "parameter_list.h"
#include "optimizer.h"

void	ParameterList::add(Variable& parameter)
{
	_parameters.push_back(&parameter);
}

void	ParameterList::step(Optimizer& optimizer)
{
	size_t i;

	i = 0;
	while (i < _parameters.size())
	{
		optimizer.step(*_parameters[i]);
		i++;
	}
}

void	ParameterList::zero_grad(Optimizer& optimizer)
{
	size_t i;

	i = 0;
	while (i < _parameters.size())
	{
		optimizer.zero_grad(*_parameters[i]);
		i++;
	}
}

size_t	ParameterList::size() const
{
	return (_parameters.size());
}
