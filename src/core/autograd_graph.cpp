#include "autograd_graph.h"

#include "autograd.h"
#include "math.h"

#include <stdexcept>

AutogradGraph::Node::Node(
	OperationType operation,
	Variable *output_variable,
	Variable *left_variable,
	Variable *right_variable)
	: type(operation),
	output(output_variable),
	left(left_variable),
	right(right_variable)
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
		delete _nodes[i];
		i++;
	}
}


