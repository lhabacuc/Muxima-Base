#ifndef PARAMETER_LIST_H
# define PARAMETER_LIST_H

# include "../core/variable.h"

# include <vector>

class ParameterList
{
	private:
		std::vector<Variable*>	_parameters;

	public:
		void	add(Variable& parameter);

		void	step(class Optimizer& optimizer);
		void	zero_grad(class Optimizer& optimizer);

		size_t	size() const;
		size_t	total_parameters() const;
};

#endif
