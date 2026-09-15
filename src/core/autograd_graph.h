#ifndef AUTOGRAD_GRAPH_H
# define AUTOGRAD_GRAPH_H

# include "variable.h"

# include <vector>

class AutogradGraph
{
	private:
		enum OperationType
		{
			OP_NONE,
			OP_ADD,
			OP_SUBTRACT,
			OP_MULTIPLY,
			OP_MATMUL
		};

		struct Node
		{
			OperationType	type;
			Variable		*output;
			Variable		*left;
			Variable		*right;

			Node(
				OperationType type,
				Variable *output,
				Variable *left,
				Variable *right);
		};

		std::vector<Node*>	_nodes;

		Node*	find_node(
			Variable& variable);

		void	backward_node(
			Node *node);

		void	build_topology(
			Variable& variable,
			std::vector<Node*>& topology,
			std::vector<Variable*>& visited);

		void	build_node_topology(
			Node *node,
			std::vector<Node*>& topology,
			std::vector<Variable*>& visited);

	public:
		AutogradGraph();
		~AutogradGraph();

		Variable*	add(
			Variable& left,
			Variable& right);

		Variable*	subtract(
			Variable& left,
			Variable& right);

		Variable*	multiply(
			Variable& left,
			Variable& right);

		Variable*	matmul(
			Variable& left,
			Variable& right);

		void	backward(
			Variable& output);
};

#endif
