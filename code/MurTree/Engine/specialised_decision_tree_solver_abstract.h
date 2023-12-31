﻿//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#pragma once

//#include "specialised_branch_misclassification_computer.h"
#include "../Data Structures/binary_data.h"
#include "../Data Structures/decision_tree.h"
#include "../Data Structures/children_information.h"
#include "../Data Structures/internal_node_description.h"

#include <vector>

namespace MurTree
{
struct SpecialisedSolverNode
{
	SpecialisedSolverNode(double inici) {
		error = inici;
		feature = num_nodes_left = num_nodes_right = INT32_MAX;
	}

	SpecialisedSolverNode() { Initialise(); }
	void Initialise() {
		error = DBL_MAX;
		num_nodes_right = num_nodes_left = feature = INT32_MAX;
	}

	double Error() const { return error; }

	uint32_t feature;
	double error;
	uint32_t num_nodes_left, num_nodes_right;
};

struct SpecialisedDecisionTreeSolverResult
{
	SpecialisedSolverNode node_budget_one, node_budget_two, node_budget_three;

	SpecialisedDecisionTreeSolverResult() {}

	SpecialisedDecisionTreeSolverResult(double error) :
		node_budget_one(error),
		node_budget_two(error),
		node_budget_three(error)
	{
	}
};

struct SpecialisedDecisionTreeSolverResult2
{
	InternalNodeDescription node_budget_one, node_budget_two, node_budget_three;

	SpecialisedDecisionTreeSolverResult2() :
		node_budget_one(INT32_MAX, INT32_MAX, DBL_MAX, INT32_MAX, INT32_MAX),
		node_budget_two(INT32_MAX, INT32_MAX, DBL_MAX, INT32_MAX, INT32_MAX),
		node_budget_three(INT32_MAX, INT32_MAX, DBL_MAX, INT32_MAX, INT32_MAX)
	{}
};

class SpecialisedDecisionTreeSolverAbstract
{
public:
	virtual SpecialisedDecisionTreeSolverResult2 Solve(BinaryDataInternal& data) = 0;
	virtual int ProbeDifference(BinaryDataInternal& data) = 0;
};
}