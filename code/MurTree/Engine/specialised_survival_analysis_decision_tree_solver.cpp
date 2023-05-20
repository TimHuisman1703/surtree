//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#include "specialised_survival_analysis_decision_tree_solver.h"
#include "binary_data_difference_computer.h"
#include "solver.h"
#include "../Utilities/runtime_assert.h"

#include <time.h>

namespace MurTree
{
SpecialisedSurvivalAnalaysisDecisionTreeSolver::SpecialisedSurvivalAnalaysisDecisionTreeSolver(int num_features, int num_feature_vectors, bool use_incremental_frequencies) :
	num_features_(num_features),
	penalty_computer_(num_features, num_feature_vectors, use_incremental_frequencies),
	best_children_error_info_(num_features),
	time_loop(0),
	time_initi(0)
{
}

SpecialisedDecisionTreeSolverResult2 MurTree::SpecialisedSurvivalAnalaysisDecisionTreeSolver::Solve(BinaryDataInternal& data)
{	
	bool changes_made = Initialise(data);
	if (!changes_made) { return previous_output_; }
	ComputeOptimalTreesBasedOnFrequencyCounts();
	previous_output_ = ConstructOutput();
	return previous_output_;
}

int SpecialisedSurvivalAnalaysisDecisionTreeSolver::ProbeDifference(BinaryDataInternal& data)
{
	return penalty_computer_.ProbeDifference(data);
}

bool SpecialisedSurvivalAnalaysisDecisionTreeSolver::Initialise(BinaryDataInternal& data)
{
	clock_t clock_start = clock();
	
	bool changed_made = penalty_computer_.Initialise(data);
	if (!changed_made) { return false; }
	
	for (int i = 0; i < num_features_; i++)
	{
		best_children_error_info_[i].left_child_feature = INT32_MAX;
		best_children_error_info_[i].right_child_feature = INT32_MAX;
		best_children_error_info_[i].left_child_penalty = DBL_MAX;
		best_children_error_info_[i].right_child_penalty = DBL_MAX;
	}
	
	// TODO-SA: Add back leaf error
	results_ = SpecialisedDecisionTreeSolverResult(1e9 /*leaf_error*/);
	time_initi += double(clock() - clock_start) / CLOCKS_PER_SEC;
	return true;
}

void SpecialisedSurvivalAnalaysisDecisionTreeSolver::ComputeOptimalTreesBasedOnFrequencyCounts()
{
	clock_t clock_start = clock();
	for (int f1 = 0; f1 < num_features_; f1++)
	{
		double penalty_left_error = penalty_computer_.PenaltyBranchZeroZero(f1, f1);
		double penalty_right_error = penalty_computer_.PenaltyBranchOneOne(f1, f1);

		//update the misclassification for the tree with only one node
		double penalty_one_node = (penalty_left_error + penalty_right_error);

		if (penalty_one_node < results_.node_budget_one.Error())
		{
			results_.node_budget_one.feature = f1;
			results_.node_budget_one.error = (penalty_left_error + penalty_right_error);
			results_.node_budget_one.num_nodes_left = 0;
			results_.node_budget_one.num_nodes_right = 0;
		}

		for (int f2 = f1 + 1; f2 < num_features_; f2++)
		{
			double penalty_branch_one_one = penalty_computer_.PenaltyBranchOneOne(f1, f2);
			double penalty_branch_one_zero = penalty_computer_.PenaltyBranchOneZero(f1, f2);
			double penalty_branch_zero_one = penalty_computer_.PenaltyBranchZeroOne(f1, f2);
			double penalty_branch_zero_zero = penalty_computer_.PenaltyBranchZeroZero(f1, f2);

			UpdateBestLeftChild(f1, f2, penalty_branch_zero_one + penalty_branch_zero_zero);
			UpdateBestRightChild(f1, f2, penalty_branch_one_one + penalty_branch_one_zero);

			UpdateBestLeftChild(f2, f1, penalty_branch_one_zero + penalty_branch_zero_zero);
			UpdateBestRightChild(f2, f1, penalty_branch_one_one + penalty_branch_zero_one);

			//update the best tree with two nodes in case a better tree has been found
			//use f1 as root
			UpdateBestTwoNodeAssignment(f1, penalty_left_error, penalty_right_error);
			//use f2 as root
			UpdateBestTwoNodeAssignment(f2, penalty_computer_.PenaltyBranchZeroZero(f2, f2), penalty_computer_.PenaltyBranchOneOne(f2, f2));
		}
		UpdateThreeNodeTreeInfo(f1); //it is important to call this after the previous loop, i.e., after calling UpdateTwoNodeTreeInfo(f1, f2) for all f2 > f1
	}
	time_loop += double(clock() - clock_start) / CLOCKS_PER_SEC;
}

SpecialisedDecisionTreeSolverResult2 SpecialisedSurvivalAnalaysisDecisionTreeSolver::ConstructOutput()
{
	SpecialisedDecisionTreeSolverResult2 result_final;
	result_final.node_budget_one.feature = results_.node_budget_one.feature;
	result_final.node_budget_one.error = results_.node_budget_one.error;
	result_final.node_budget_one.num_nodes_left = results_.node_budget_one.num_nodes_left;
	result_final.node_budget_one.num_nodes_right = results_.node_budget_one.num_nodes_right;
	result_final.node_budget_one.theta = 0;

	result_final.node_budget_two.feature = results_.node_budget_two.feature;
	result_final.node_budget_two.error = results_.node_budget_two.error;
	result_final.node_budget_two.num_nodes_left = results_.node_budget_two.num_nodes_left;
	result_final.node_budget_two.num_nodes_right = results_.node_budget_two.num_nodes_right;
	result_final.node_budget_two.theta = 0;

	result_final.node_budget_three.feature = results_.node_budget_three.feature;
	result_final.node_budget_three.error = results_.node_budget_three.error;
	result_final.node_budget_three.num_nodes_left = results_.node_budget_three.num_nodes_left;
	result_final.node_budget_three.num_nodes_right = results_.node_budget_three.num_nodes_right;
	result_final.node_budget_three.theta = 0;

	return result_final;
}

void SpecialisedSurvivalAnalaysisDecisionTreeSolver::UpdateBestTwoNodeAssignment(int root_feature, double error_left, double error_right)
{
	double objective_two_nodes = best_children_error_info_[root_feature].left_child_penalty + error_right;
	if (results_.node_budget_two.Error() > objective_two_nodes)
	{
		results_.node_budget_two.error = objective_two_nodes;
		results_.node_budget_two.feature = root_feature;
		results_.node_budget_two.num_nodes_left = 1;
		results_.node_budget_two.num_nodes_right = 0;
	}

	objective_two_nodes = best_children_error_info_[root_feature].right_child_penalty + error_left;
	if (results_.node_budget_two.Error() > objective_two_nodes)
	{
		results_.node_budget_two.error = objective_two_nodes;
		results_.node_budget_two.feature = root_feature;
		results_.node_budget_two.num_nodes_left = 0;
		results_.node_budget_two.num_nodes_right = 1;
	}
}

void SpecialisedSurvivalAnalaysisDecisionTreeSolver::UpdateThreeNodeTreeInfo(int root_feature)
{
	runtime_assert(best_children_error_info_[root_feature].left_child_penalty != DBL_MAX);
	runtime_assert(best_children_error_info_[root_feature].right_child_penalty != DBL_MAX);

	double new_penalty = best_children_error_info_[root_feature].left_child_penalty + best_children_error_info_[root_feature].right_child_penalty;
	if (results_.node_budget_three.Error() > new_penalty)
	{
		results_.node_budget_three.error = new_penalty;
		results_.node_budget_three.feature = root_feature;
		results_.node_budget_three.num_nodes_left = 1;
		results_.node_budget_three.num_nodes_right = 1;
	}
}

void SpecialisedSurvivalAnalaysisDecisionTreeSolver::UpdateBestLeftChild(int feature, int left_child_feature, double penalty)
{
	if (best_children_error_info_[feature].left_child_penalty > penalty)
	{
		best_children_error_info_[feature].left_child_feature = left_child_feature;
		best_children_error_info_[feature].left_child_penalty = penalty;
	}
}

void SpecialisedSurvivalAnalaysisDecisionTreeSolver::UpdateBestRightChild(int feature, int right_child_feature, double penalty)
{
	if (best_children_error_info_[feature].right_child_penalty > penalty)
	{
		best_children_error_info_[feature].right_child_feature = right_child_feature;
		best_children_error_info_[feature].right_child_penalty = penalty;
	}
}

}//end namespace MurTree