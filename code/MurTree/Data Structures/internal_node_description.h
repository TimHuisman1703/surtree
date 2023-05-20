//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#pragma once

#include "../Data Structures/binary_data.h"

//feature, objective value, num_nodes_left, num_nodes_right
namespace MurTree
{
	struct InternalNodeDescription
	{
		int feature;
		double theta;
		double error;
		int num_nodes_left;
		int num_nodes_right;

		InternalNodeDescription(int feature, double theta, double error, int num_nodes_left, int num_nodes_right);

		double Error() const;
		double SparseObjective(double sparse_coefficient) const;
		int NumNodes() const;
		bool IsInfeasible() const;
		bool IsFeasible() const;
		
		static InternalNodeDescription CreateInfeasibleNodeDescription();
	};
}