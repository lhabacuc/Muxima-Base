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

Variable*	AutogradGraph::add(
	Variable& left,
	Variable& right)
{
	Tensor	value;
	Variable	*output;
	Node		*node;

	value = ::add(left.value(), right.value());
	output = new Variable(value, true);

	node = new Node(
		OP_ADD,
		output,
		&left,
		&right);

	_nodes.push_back(node);
	return (output);
}

Variable*	AutogradGraph::subtract(
	Variable& left,
	Variable& right)
{
	Tensor	value;
	Variable	*output;
	Node		*node;

	value = ::subtract(left.value(), right.value());
	output = new Variable(value, true);

	node = new Node(
		OP_SUBTRACT,
		output,
		&left,
		&right);

	_nodes.push_back(node);
	return (output);
}

Variable*	AutogradGraph::multiply(
	Variable& left,
	Variable& right)
{
	Tensor	value;
	Variable	*output;
	Node		*node;

	value = ::multiply(left.value(), right.value());
	output = new Variable(value, true);

	node = new Node(
		OP_MULTIPLY,
		output,
		&left,
		&right);

	_nodes.push_back(node);
	return (output);
}

Variable*	AutogradGraph::matmul(
	Variable& left,
	Variable& right)
{
	Tensor	value;
	Variable	*output;
	Node		*node;

	value = ::matmul(left.value(), right.value());
	output = new Variable(value, true);

	node = new Node(
		OP_MATMUL,
		output,
		&left,
		&right);

	_nodes.push_back(node);
	return (output);
}

void	AutogradGraph::backward_node(Node *node)
{
	if (node->type == OP_ADD)
	{
		Autograd::add_backward(
			*node->left,
			*node->right,
			node->output->gradient());
	}
	else if (node->type == OP_SUBTRACT)
	{
		Autograd::subtract_backward(
			*node->left,
			*node->right,
			node->output->gradient());
	}
	else if (node->type == OP_MULTIPLY)
	{
		Autograd::multiply_backward(
			*node->left,
			*node->right,
			node->output->gradient());
	}
	else if (node->type == OP_MATMUL)
	{
		Autograd::matmul_backward(
			*node->left,
			*node->right,
			node->output->gradient());
	}
}

void	AutogradGraph::backward(Variable& output)
{
	size_t	i;

	if (!output.requires_grad())
		return ;

	i = 0;
	while (i < output.gradient().size())
	{
		output.gradient().data()[i] = 1.0f;
		i++;
	}

	i = _nodes.size();

	while (i > 0)
	{
		i--;

		if (_nodes[i]->output == &output
			|| _nodes[i]->output->requires_grad())
			backward_node(_nodes[i]);
	}
}
