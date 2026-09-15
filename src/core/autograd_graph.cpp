#include "autograd_graph.h"

#include "autograd.h"
#include "math.h"

#include "../training/cross_entropy.h"

#include <cstddef>

AutogradGraph::Node::Node(
	OperationType operation,
	Variable *output_variable,
	Variable *left_variable,
	Variable *right_variable,
	Tensor *target_tensor)
	: type(operation),
	output(output_variable),
	left(left_variable),
	right(right_variable),
	targets(target_tensor)
{
}

AutogradGraph::AutogradGraph()
{
}

AutogradGraph::~AutogradGraph()
{
	size_t	i;

	i = 0;
	while (i < _nodes.size())
	{
		delete _nodes[i]->output;
		delete _nodes[i]->targets;
		delete _nodes[i];
		i++;
	}
}

AutogradGraph::Node*	AutogradGraph::find_node(
	Variable& variable)
{
	size_t	i;

	i = 0;
	while (i < _nodes.size())
	{
		if (_nodes[i]->output == &variable)
			return (_nodes[i]);
		i++;
	}
	return (NULL);
}

Variable*	AutogradGraph::add(
	Variable& left,
	Variable& right)
{
	Tensor		value = ::add(
		left.value(),
		right.value());
	Variable	*output;
	Node		*node;

	output = new Variable(value, true);

	node = new Node(
		OP_ADD,
		output,
		&left,
		&right,
		NULL);

	_nodes.push_back(node);
	return (output);
}

Variable*	AutogradGraph::subtract(
	Variable& left,
	Variable& right)
{
	Tensor		value = ::subtract(
		left.value(),
		right.value());
	Variable	*output;
	Node		*node;

	output = new Variable(value, true);

	node = new Node(
		OP_SUBTRACT,
		output,
		&left,
		&right,
		NULL);

	_nodes.push_back(node);
	return (output);
}

Variable*	AutogradGraph::multiply(
	Variable& left,
	Variable& right)
{
	Tensor		value = ::multiply(
		left.value(),
		right.value());
	Variable	*output;
	Node		*node;

	output = new Variable(value, true);

	node = new Node(
		OP_MULTIPLY,
		output,
		&left,
		&right,
		NULL);

	_nodes.push_back(node);
	return (output);
}

Variable*	AutogradGraph::matmul(
	Variable& left,
	Variable& right)
{
	Tensor		value = ::matmul(
		left.value(),
		right.value());
	Variable	*output;
	Node		*node;

	output = new Variable(value, true);

	node = new Node(
		OP_MATMUL,
		output,
		&left,
		&right,
		NULL);

	_nodes.push_back(node);
	return (output);
}

Variable*	AutogradGraph::relu(
	Variable& input)
{
	Tensor		value = ::relu(input.value());
	Variable	*output;
	Node		*node;

	output = new Variable(
		value,
		true);

	node = new Node(
		OP_RELU,
		output,
		&input,
		NULL,
		NULL);

	_nodes.push_back(node);

	return (output);
}

Variable*	AutogradGraph::gelu(
	Variable& input)
{
	Tensor		value = ::gelu(input.value());
	Variable	*output;
	Node		*node;

	output = new Variable(
		value,
		true);

	node = new Node(
		OP_GELU,
		output,
		&input,
		NULL,
		NULL);

	_nodes.push_back(node);

	return (output);
}

Variable*	AutogradGraph::cross_entropy(
	Variable& logits,
	const Tensor& targets)
{
	Tensor		value_tensor(
		std::vector<size_t>(1, 1));
	Variable	*output;
	Tensor		*stored_targets;
	Node		*node;

	value_tensor.data()[0] =
		CrossEntropy::forward(
			logits.value(),
			targets);

	output = new Variable(
		value_tensor,
		true);

	stored_targets = new Tensor(targets);

	node = new Node(
		OP_CROSS_ENTROPY,
		output,
		&logits,
		NULL,
		stored_targets);

	_nodes.push_back(node);

	return (output);
}

void	AutogradGraph::backward_node(
	Node *node)
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
	else if (node->type == OP_RELU)
	{
		Autograd::relu_backward(
			*node->left,
			node->output->gradient());
	}
	else if (node->type == OP_GELU)
	{
		Autograd::gelu_backward(
			*node->left,
			node->output->gradient());
	}
	else if (node->type == OP_CROSS_ENTROPY)
	{
		Autograd::cross_entropy_backward(
			*node->left,
			*node->targets,
			node->output->gradient());
	}
}

void	AutogradGraph::build_node_topology(
	Node *node,
	std::vector<Node*>& topology,
	std::vector<Variable*>& visited)
{
	size_t	i;
	Node	*parent;

	i = 0;
	while (i < visited.size())
	{
		if (visited[i] == node->output)
			return ;
		i++;
	}

	visited.push_back(node->output);

	parent = find_node(*node->left);
	if (parent != NULL)
	{
		build_node_topology(
			parent,
			topology,
			visited);
	}

	if (node->right != NULL)
	{
		parent = find_node(*node->right);
		if (parent != NULL)
		{
			build_node_topology(
				parent,
				topology,
				visited);
		}
	}

	topology.push_back(node);
}

void	AutogradGraph::build_topology(
	Variable& variable,
	std::vector<Node*>& topology,
	std::vector<Variable*>& visited)
{
	Node	*node;

	node = find_node(variable);
	if (node == NULL)
		return ;

	build_node_topology(
		node,
		topology,
		visited);
}

void	AutogradGraph::backward(
	Variable& output)
{
	std::vector<Node*>		topology;
	std::vector<Variable*>	visited;
	size_t					i;

	if (!output.requires_grad())
		return ;

	i = 0;
	while (i < output.gradient().size())
	{
		output.gradient().data()[i] = 1.0f;
		i++;
	}

	build_topology(
		output,
		topology,
		visited);

	i = topology.size();

	while (i > 0)
	{
		i--;

		backward_node(
			topology[i]);
	}
}
