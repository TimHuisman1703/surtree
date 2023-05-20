//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#pragma once

#include "specialised_decision_tree_solver_abstract.h"
#include "specialised_branch_survival_analysis_computer.h"
#include "../Data Structures/binary_data.h"
#include "../Data Structures/decision_tree.h"
#include "../Data Structures/children_information.h"
#include "../Data Structures/internal_node_description.h"

#include <vector>

namespace MurTree
{
class SpecialisedSurvivalAnalaysisDecisionTreeSolver : public SpecialisedDecisionTreeSolverAbstract
{
public:
	//assumes the feature vectors ID are in the range of [0, ..., num_feature_vectors)
	SpecialisedSurvivalAnalaysisDecisionTreeSolver(int num_features, int num_feature_vectors, bool use_incremental_frequencies); 
	 
	SpecialisedDecisionTreeSolverResult2 Solve(BinaryDataInternal& data);
	int ProbeDifference(BinaryDataInternal& data);

	double time_initi, time_loop;

private:
	int num_features_;
	SpecialisedDecisionTreeSolverResult results_;
	SpecialisedDecisionTreeSolverResult2 previous_output_;

	std::vector<ChildrenInformation> best_children_error_info_;

	bool Initialise(BinaryDataInternal& data); //initialises internal data structures, including frequency counts. Returns true if any changes were made. This is useful since if no changes were made, we will reuse the previous solution.
	void ComputeOptimalTreesBasedOnFrequencyCounts();
	SpecialisedDecisionTreeSolverResult2 ConstructOutput();

	void UpdateBestTwoNodeAssignment(int root_feature, double error_left, double error_right);
	void UpdateThreeNodeTreeInfo(int feature);
	void UpdateBestLeftChild(int feature, int left_child_feature, double penalty);
	void UpdateBestRightChild(int feature, int right_child_feature, double penalty);

	SpecialisedBranchSurvivalAnalysisComputer penalty_computer_;
};
}//end namespace MurTree