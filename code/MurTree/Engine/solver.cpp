﻿//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#include "solver.h"
#include "specialised_survival_analysis_decision_tree_solver.h"
#include "branch_cache.h"
#include "dataset_cache.h"
#include "cache_closure.h"
#include "feature_selector_in_order.h"
#include "feature_selector_random.h"
#include "nelson_aalen_estimator.h"
#include "cache_closure.h"
#include "../Utilities/runtime_assert.h"
#include "../Utilities/file_reader.h"
#include "../Data Structures/child_subtree_info.h"

namespace MurTree
{
Solver::Solver(ParameterHandler& parameters):
	verbose_(parameters.GetBooleanParameter("verbose")),
	cache_(0),
	binary_data_(0),
	feature_selectors_(100, 0),
	splits_data(100, 0),
	specialised_solver1_(0),
	specialised_solver2_(0)
{
	if (parameters.GetStringParameter("node-selection") == "dynamic") { dynamic_child_selection_ = true; }
	else if (parameters.GetStringParameter("node-selection") == "post-order") { dynamic_child_selection_ = false; }
	else { std::cout << "Unknown node selection strategy: '" << parameters.GetStringParameter("node-selection") << "'\n"; exit(1); }

	//start: read in the data
	feature_vectors_ = FileReader::ReadDataDL(parameters.GetStringParameter("file"), (int)parameters.GetIntegerParameter("duplicate-factor"));
	num_features_ = -1;
	if (!feature_vectors_.empty()) { num_features_ = feature_vectors_[0].NumTotalFeatures(); } //could do better checking of the data
	runtime_assert(num_features_ > 0);

	binary_data_ = new BinaryDataInternal(num_features_);
	for (int i = 0; i < feature_vectors_.size(); i++)
	{
		binary_data_->AddFeatureVector(&feature_vectors_[i]);
	}
	//end: read in the data

	for(int i = 0; i < 100; i++) { splits_data[i] = new SplitBinaryData(num_features_); } 
	
	for (int i = 0; i < 100; i++) 
	{ 
		if (parameters.GetStringParameter("feature-ordering") == "in-order") { feature_selectors_[i] = new FeatureSelectorInOrder(num_features_); }
		else if (parameters.GetStringParameter("feature-ordering") == "random") { feature_selectors_[i] = new FeatureSelectorRandom(num_features_); }
		else { std::cout << "Unknown feature ordering strategy!\n"; exit(1); }
	}

	if (parameters.GetStringParameter("cache-type") == "branch") { cache_ = new BranchCache(100); }
	else if (parameters.GetStringParameter("cache-type") == "dataset") { cache_ = new DatasetCache(binary_data_->Size()); }
	else if (parameters.GetStringParameter("cache-type") == "closure") { cache_ = new ClosureCache(num_features_, binary_data_->Size()); }
	else
	{
		std::cout << "Parameter error: unknown cache type: " << parameters.GetStringParameter("cache-type") << "\n";
		runtime_assert(1 == 2);
	}

	specialised_solver1_ = new SpecialisedSurvivalAnalaysisDecisionTreeSolver
	(
		num_features_,
		binary_data_->Size(),
		parameters.GetBooleanParameter("incremental-frequency")
	);

	specialised_solver2_ = new SpecialisedSurvivalAnalaysisDecisionTreeSolver
	(
		num_features_,
		binary_data_->Size(),
		parameters.GetBooleanParameter("incremental-frequency")
	);
}

Solver::~Solver()
{
	delete cache_;
	delete specialised_solver1_;
	delete specialised_solver2_;
	delete binary_data_;
	for (int i = 0; i < splits_data.size(); i++) { delete splits_data[i]; }
	for (int i = 0; i < feature_selectors_.size(); i++) { delete feature_selectors_[i]; }
}

void Solver::ReplaceData(std::vector<FeatureVectorBinary>& new_instances)
{
	feature_vectors_ = new_instances;
	binary_data_->Clear();
	for (int i = 0; i < feature_vectors_.size(); i++)
	{
		binary_data_->AddFeatureVector(&feature_vectors_[i]);
	}
}

SolverResult Solver::Solve(ParameterHandler& parameters)
{
	stopwatch_.Initialise(parameters.GetFloatParameter("time"));

	double sparse_coefficient = parameters.GetFloatParameter("sparse-coefficient") * binary_data_->Size();
	NelsonAalenEstimator nae = { *binary_data_ };
	Branch root_branch;
	InternalNodeDescription best_solution = CreateLeafNodeDescription(*binary_data_);
	int min_num_nodes = (int)parameters.GetIntegerParameter("max-num-nodes");
	int max_num_nodes = (int)parameters.GetIntegerParameter("max-num-nodes");
	if (parameters.GetBooleanParameter("all-trees") || sparse_coefficient > 0) { min_num_nodes = 1; }
	if (verbose_) std::cout << "Leaf value: " << best_solution.Error() << "\n";
	for (int num_nodes = min_num_nodes; num_nodes <= max_num_nodes; num_nodes++)
	{
		if (!stopwatch_.IsWithinTimeLimit()) { break; }

		if (verbose_) std::cout << "num nodes: " << num_nodes << " " << stopwatch_.TimeElapsedInSeconds() << "s" << std::endl;

		int max_depth = std::min(int(parameters.GetIntegerParameter("max-depth")), num_nodes);
		double error_upper_bound = std::min(best_solution.SparseObjective(sparse_coefficient) - sparse_coefficient * num_nodes, parameters.GetFloatParameter("upper-bound") - sparse_coefficient * num_nodes);
		InternalNodeDescription current_best = SolveSubtree(
			*binary_data_,
			root_branch,
			max_depth,
			num_nodes,
			error_upper_bound
		);

		runtime_assert(current_best.IsInfeasible() || current_best.SparseObjective(sparse_coefficient) <= best_solution.SparseObjective(sparse_coefficient));

		if (current_best.IsInfeasible()) {
			if (verbose_) std::cout << "Infeasible, terminating search\n";
			break;
		}

		runtime_assert(current_best.SparseObjective(sparse_coefficient) <= best_solution.SparseObjective(sparse_coefficient));
		if (verbose_) std::cout << "Tree with " << num_nodes << " nodes: error = " << current_best.error << "; time = " << stopwatch_.TimeElapsedInSeconds() << "\n";
		best_solution = current_best;
	}

	SolverResult result;
	result.error = best_solution.Error();
	result.is_proven_optimal = stopwatch_.IsWithinTimeLimit();
	result.decision_tree_ = stopwatch_.IsWithinTimeLimit() ? ConstructOptimalTree(*binary_data_, root_branch, std::min(int(parameters.GetIntegerParameter("max-depth")), best_solution.NumNodes()), best_solution.NumNodes()) : 0;
	
	if (verbose_)
	{
		std::cout << "time: " << stopwatch_.TimeElapsedInSeconds() << "\n";
		std::cout << "Successfully terminated: " << result.is_proven_optimal << "\n";
		std::cout << "new Terminal time: " << stats_.time_in_terminal_node << "\n";
		std::cout << "Terminal calls: " << stats_.num_terminal_nodes_with_node_budget_one + stats_.num_terminal_nodes_with_node_budget_two + stats_.num_terminal_nodes_with_node_budget_three << "\n";
		std::cout << "\tTerminal 1node: " << stats_.num_terminal_nodes_with_node_budget_one << "\n";
		std::cout << "\tTerminal 2node: " << stats_.num_terminal_nodes_with_node_budget_two << "\n";
		std::cout << "\tTerminal 3node: " << stats_.num_terminal_nodes_with_node_budget_three << "\n";
		//std::cout << "new Time looping: " << (specialised_solver1_->time_loop + specialised_solver2_->time_loop) << "\n";//specialised_solver_->time_loop << "\n";
		//std::cout << "new Time inicialising: " << (specialised_solver1_->time_initi + specialised_solver2_->time_initi) << "\n";//specialised_solver_->time_initi << "\n";
	}

	DecisionNode* root_node = DecisionNode::CreateLabelNode(1.0);
	double base_error = result.is_proven_optimal ? root_node->ComputeError(*binary_data_) : -1;
	double recomputed_error = result.is_proven_optimal ? result.decision_tree_->ComputeError(*binary_data_) : -1;

	if (verbose_)
	{
		std::cout << "RECOMPUTED ERROR = " << recomputed_error << "\n";
		std::cout << "RATIO SCORE = " << std::max(0.0, 1.0 - recomputed_error / base_error) << "\n";
	}
	if (std::abs(result.error - recomputed_error) < 1e-6)
	{
		if (verbose_) std::cout << "Tree misclassification score has been verified!\n";
	}
	else if (std::abs(result.error - recomputed_error) > 1e-6 && result.is_proven_optimal)
	{
		std::cout << "Problem: algorithm reported optimal solution with an error of " << result.error << ", but tree gives " << recomputed_error << "\n";
		std::cout << "Please report this issue to Emir Demirović, e.demirovic@tudelft.nl\n";
		runtime_assert(1 == 2);
	}

	if (verbose_) std::cout << "Cache entries: " << cache_->NumEntries() << "\n";

	return result;// ConstructOutput();
}

void Solver::SetVerbosity(bool verbose)
{
	verbose_ = verbose;
}

DecisionNode* Solver::ConstructOptimalTree(BinaryDataInternal& data, Branch& branch, int max_depth, int num_nodes)
{
	runtime_assert(num_nodes >= 0);

	//check if the node meets the basic leaf criteria
	double branch_lower_bound = cache_->RetrieveLowerBound(data, branch, max_depth, num_nodes);
	if (max_depth == 0 || num_nodes == 0 || std::abs(LeafError(data) - branch_lower_bound) < 1e-6)
	{
		double theta = LeafTheta(data);
		return DecisionNode::CreateLabelNode(theta);
	}
	//recover the optimal assignment from cache - however note that not all nodes are in the cache, see next IF
	else if (cache_->IsOptimalAssignmentCached(data, branch, max_depth, num_nodes))
	{
		InternalNodeDescription optimal_node = cache_->RetrieveOptimalAssignment(data, branch, max_depth, num_nodes);
		DecisionNode* feature_node = DecisionNode::CreateFeatureNodeWithNullChildren(optimal_node.feature);
		SplitBinaryData split_data(data.NumFeatures());

		split_data.SplitData(optimal_node.feature, data);

		Branch left_branch = Branch::LeftChildBranch(branch, optimal_node.feature);
		Branch right_branch = Branch::RightChildBranch(branch, optimal_node.feature);

		int left_depth = std::min(max_depth - 1, optimal_node.num_nodes_left);
		int right_depth = std::min(max_depth - 1, optimal_node.num_nodes_right);
		DecisionNode* left_child = ConstructOptimalTree(split_data.data_without_feature, left_branch, left_depth, optimal_node.num_nodes_left);
		DecisionNode* right_child = ConstructOptimalTree(split_data.data_with_feature, right_branch, right_depth, optimal_node.num_nodes_right);

		feature_node->left_child_ = left_child;
		feature_node->right_child_ = right_child;

		return feature_node;
	}
	else if (IsTerminalNode(max_depth, num_nodes))
	{
		//actually this code is called to compute the num_nodes = 1 case
		runtime_assert(num_nodes == 1);

		runtime_assert(num_nodes == 1 || num_nodes == 2);
		//this will disappear in future versions of the code
		//terminal nodes are cached normally
		//here some nodes may not be cached -> will be in the future
		SpecialisedDecisionTreeSolverResult2 results = specialised_solver1_->Solve(data);

		if (num_nodes == 1 && std::abs(results.node_budget_one.error - LeafError(data)) < 1e-6
			|| num_nodes == 2 && std::abs(results.node_budget_two.error - LeafError(data)) < 1e-6)
		{
			return DecisionNode::CreateLabelNode(LeafTheta(data));
		}

		DecisionNode* feature_node;
		SplitBinaryData split_data(data.NumFeatures());
		
		if (num_nodes == 1)
		{
			feature_node = DecisionNode::CreateFeatureNodeWithNullChildren(results.node_budget_one.feature);
			split_data.SplitData(results.node_budget_one.feature, data);

			Branch left_branch = Branch::LeftChildBranch(branch, results.node_budget_one.feature);
			Branch right_branch = Branch::RightChildBranch(branch, results.node_budget_one.feature);

			int left_depth = std::min(max_depth - 1, results.node_budget_one.num_nodes_left);
			int right_depth = std::min(max_depth - 1, results.node_budget_one.num_nodes_right);
			DecisionNode* left_child = ConstructOptimalTree(split_data.data_without_feature, left_branch, left_depth, results.node_budget_one.num_nodes_left);
			DecisionNode* right_child = ConstructOptimalTree(split_data.data_with_feature, right_branch, right_depth, results.node_budget_one.num_nodes_right);

			feature_node->left_child_ = left_child;
			feature_node->right_child_ = right_child;

			return feature_node;
		}
		else
		{
			feature_node = DecisionNode::CreateFeatureNodeWithNullChildren(results.node_budget_two.feature);
			split_data.SplitData(results.node_budget_two.feature, data);

			Branch left_branch = Branch::LeftChildBranch(branch, results.node_budget_two.feature);
			Branch right_branch = Branch::RightChildBranch(branch, results.node_budget_two.feature);

			int left_depth = std::min(max_depth - 1, results.node_budget_two.num_nodes_left);
			int right_depth = std::min(max_depth - 1, results.node_budget_two.num_nodes_right);
			DecisionNode* left_child = ConstructOptimalTree(split_data.data_without_feature, left_branch, left_depth, results.node_budget_two.num_nodes_left);
			DecisionNode* right_child = ConstructOptimalTree(split_data.data_with_feature, right_branch, right_depth, results.node_budget_two.num_nodes_right);

			feature_node->left_child_ = left_child;
			feature_node->right_child_ = right_child;

			return feature_node;
		}		
	}
	else
	{
		runtime_assert(1 == 3); //I think this is no longer used

		if (max_depth != 1) { std::cout << "OBVIOUSLY WRONG\n"; return DecisionNode::CreateLabelNode(LeafTheta(data)); }

		//the remaining nodes that got to this IF statement are children of terminal nodes that did not meet the basic leaf criteria
		runtime_assert(max_depth == 1);
	
		SpecialisedDecisionTreeSolverResult2 results = specialised_solver1_->Solve(data);
		
		if (std::abs(results.node_budget_one.error - LeafError(data)) < 1e-4)
		{
			return DecisionNode::CreateLabelNode(LeafTheta(data));
		}
		
		DecisionNode* feature_node = DecisionNode::CreateFeatureNodeWithNullChildren(results.node_budget_one.feature);
		SplitBinaryData split_data(data.NumFeatures());
		split_data.SplitData(results.node_budget_one.feature, data);

		Branch left_branch = Branch::LeftChildBranch(branch, results.node_budget_one.feature);
		Branch right_branch = Branch::RightChildBranch(branch, results.node_budget_one.feature);

		int left_depth = std::min(max_depth - 1, results.node_budget_two.num_nodes_left);
		int right_depth = std::min(max_depth - 1, results.node_budget_two.num_nodes_right);
		DecisionNode* left_child = ConstructOptimalTree(split_data.data_without_feature, left_branch, left_depth, results.node_budget_two.num_nodes_left);
		DecisionNode* right_child = ConstructOptimalTree(split_data.data_with_feature, right_branch, right_depth, results.node_budget_two.num_nodes_right);

		feature_node->left_child_ = left_child;
		feature_node->right_child_ = right_child;

		return feature_node;
	}
}

InternalNodeDescription Solver::SolveSubtree(BinaryDataInternal& data, Branch& branch, int max_depth, int num_nodes, double upper_bound)
{//corresponds to Algorithm 1 from the paper
	runtime_assert(0 <= max_depth && max_depth <= num_nodes);

	if (!stopwatch_.IsWithinTimeLimit()) { return CreateInfeasibleNodeDescription(); }
	
	// Prune based on upper bound
	if (upper_bound <= 0) { return CreateInfeasibleNodeDescription(); }

	// Base case (Eq. 1), second case: no feature nodes are possible
	if (max_depth == 0 || num_nodes == 0) { return (LeafError(data) >= upper_bound ? CreateInfeasibleNodeDescription() : CreateLeafNodeDescription(data)); }
	
	// Use cached subtrees if possible (Section 4.5)
	InternalNodeDescription cached_optimal_node = cache_->RetrieveOptimalAssignment(data, branch, max_depth, num_nodes);
	if (!cached_optimal_node.IsInfeasible())
	{
		return (cached_optimal_node.Error() >= upper_bound ? CreateInfeasibleNodeDescription() : cached_optimal_node);
	}

	// Use Algorithm 4 for small trees from Section 4.3
	// Note that the specialised algorithm updates the cache
	if (IsTerminalNode(max_depth, num_nodes)) { return SolveTerminalNode(data, branch, max_depth, num_nodes, upper_bound); }

	// General (fourth) case (Eq. 1): Exhaustively search using Algorithm 2 
	return SolveSubtreeGeneralCase(data, branch, max_depth, num_nodes, upper_bound);
}

InternalNodeDescription Solver::SolveSubtreeGeneralCase(BinaryDataInternal& data, Branch& branch, int max_depth, int num_nodes, double upper_bound)
{//Algorithm 2 from the paper
	runtime_assert(max_depth <= num_nodes);

	//Use a single classification node as an initial solution
	InternalNodeDescription best_node = CreateInfeasibleNodeDescription();
	if (LeafError(data) < upper_bound) { 
		best_node = CreateLeafNodeDescription(data);
	}

	SplitBinaryData& split_data = *splits_data[max_depth];

	double lower_bound_refined = DBL_MAX; //'lower_bound_refined' refers to the refined lower bound in Eq. 16
	double branch_lower_bound = cache_->RetrieveLowerBound(data, branch, max_depth, num_nodes); //find the lower bound stored in the cache (Section 4.5.4)
	int max_size_subtree = std::min((1 << (max_depth - 1)) - 1, num_nodes - 1); //compute allowed number of nodes for child subtrees, by taking the minimum between a full tree of max_depth or the number of nodes - 1
	int min_size_subtree = num_nodes - 1 - max_size_subtree;

	feature_selectors_[max_depth]->Reset(data);
	while(feature_selectors_[max_depth]->AreThereAnyFeaturesLeft())
	{
		int splitting_feature = feature_selectors_[max_depth]->PopNextFeature();
		if (!stopwatch_.IsWithinTimeLimit()) { return CreateInfeasibleNodeDescription(); }
		// If the current best node is the optimal node, stop
		if (best_node.IsFeasible() && best_node.Error() <= branch_lower_bound) { break; }

		split_data.SplitData(splitting_feature, data);
		// Splits where one child does not have any events should be avoided
		if (split_data.events_without_feature == 0 || split_data.events_with_feature == 0) {
			continue;
		}

		Branch left_branch = Branch::LeftChildBranch(branch, splitting_feature);
		Branch right_branch = Branch::RightChildBranch(branch, splitting_feature);
		for (int left_subtree_size = min_size_subtree; left_subtree_size <= max_size_subtree; left_subtree_size++)
		{
			//in the paper this loop is presented as part of Algorithm 3

			int right_subtree_size = num_nodes - left_subtree_size - 1; //the '-1' is necessary since using the parent node counts as a node
						
			//decide on the order of child nodes
			//the result is stored in 'first_child' and 'second_child'
			//a static order would always have first_child to be the left node, whereas in dynamic this is determined on the fly
			ChildSubtreeInfo left_subtree_info(&split_data.data_without_feature, left_branch, std::min(max_depth - 1, left_subtree_size), left_subtree_size);
			ChildSubtreeInfo right_subtree_info(&split_data.data_with_feature, right_branch, std::min(max_depth - 1, right_subtree_size), right_subtree_size);
			ChildSubtreeOrdering sorted_children = GetSortedChildren(left_subtree_info, right_subtree_info);
			ChildSubtreeInfo& first_child = sorted_children.first_child;
			ChildSubtreeInfo& second_child = sorted_children.second_child;

			//Impose an upper bound that ensures that a feasible tree will have fewer misclassifications than the best tree found so far
			double first_child_upper_bound = std::min((best_node.IsFeasible() ? best_node.Error() : DBL_MAX), upper_bound)
											- cache_->RetrieveLowerBound(*second_child.binary_data, second_child.branch, second_child.depth, second_child.size);
			InternalNodeDescription first_child_solution = SolveSubtree(
				*first_child.binary_data,
				first_child.branch,
				first_child.depth,
				first_child.size,
				first_child_upper_bound
			);

			if (!stopwatch_.IsWithinTimeLimit()) { return CreateInfeasibleNodeDescription(); }

			// No need to compute the other subtree if the first_child is infeasible
			if (first_child_solution.IsInfeasible())
			{
				double local_bound = cache_->RetrieveLowerBound(*left_subtree_info.binary_data, left_branch, left_subtree_info.depth, left_subtree_size) 
									+ cache_->RetrieveLowerBound(*right_subtree_info.binary_data, right_branch, right_subtree_info.depth, right_subtree_size);
				lower_bound_refined = std::min(lower_bound_refined, local_bound);
				continue;
			}

			double second_child_upper_bound = std::min((best_node.IsFeasible() ? best_node.Error() : DBL_MAX), upper_bound)
												- first_child_solution.Error();
			InternalNodeDescription second_child_solution = SolveSubtree(
				*second_child.binary_data,
				second_child.branch,
				second_child.depth,
				second_child.size,
				second_child_upper_bound
			);

			if (!stopwatch_.IsWithinTimeLimit()) { return CreateInfeasibleNodeDescription(); }

			// If both children are feasible, update the locally best solution, and the upper bound
			if (second_child_solution.IsFeasible())
			{
				InternalNodeDescription left_child = (first_child.branch == left_branch ? first_child_solution : second_child_solution);
				InternalNodeDescription right_child = (first_child.branch == right_branch ? first_child_solution : second_child_solution);
				InternalNodeDescription current_node = CombineLeftAndRightChildren(splitting_feature, left_child, right_child);
				//this condition always holds, right?
				runtime_assert(best_node.IsInfeasible() || current_node.Error() < best_node.Error() + 1e-4);
				if (best_node.IsInfeasible() || current_node.Error() < best_node.Error() + 1e-4)
				{
					best_node = current_node;
					if (best_node.Error() == branch_lower_bound) { break; }
				}
			}
			else
			{//is infeasible
				double local_bound = cache_->RetrieveLowerBound(*left_subtree_info.binary_data, left_branch, left_subtree_info.depth, left_subtree_size) 
					+ cache_->RetrieveLowerBound(*right_subtree_info.binary_data, right_branch, right_subtree_info.depth, right_subtree_size);
				lower_bound_refined = std::min(lower_bound_refined, local_bound);
			}			
		}
	}//end for loop

	if (!stopwatch_.IsWithinTimeLimit()) { return CreateInfeasibleNodeDescription(); }

	// Cache the optimal solution
	if (best_node.IsFeasible())
	{
		runtime_assert(best_node.Error() < upper_bound + 1e-4);
		cache_->StoreOptimalBranchAssignment(data, branch, best_node, max_depth, num_nodes);
	}

	return best_node;
}

InternalNodeDescription Solver::SolveTerminalNode(BinaryDataInternal& data, Branch& branch, int max_depth, int num_nodes, double upper_bound)
{
	runtime_assert(max_depth <= 2 && 1 <= num_nodes && num_nodes <= 3 && max_depth <= num_nodes);
	runtime_assert(num_nodes != 3 || !cache_->IsOptimalAssignmentCached(data, branch, 2, 3));
	runtime_assert(num_nodes != 2 || !cache_->IsOptimalAssignmentCached(data, branch, 2, 2));
	runtime_assert(num_nodes != 1 || !cache_->IsOptimalAssignmentCached(data, branch, 1, 1));

	stats_.num_terminal_nodes_with_node_budget_one += (num_nodes == 1);
	stats_.num_terminal_nodes_with_node_budget_two += (num_nodes == 2);
	stats_.num_terminal_nodes_with_node_budget_three += (num_nodes == 3);

	clock_t clock_start = clock();
	SpecialisedDecisionTreeSolverResult2 results;
	//select the solver which is already contains frequency counts that are closest to the data
	if (specialised_solver1_->ProbeDifference(data) < specialised_solver2_->ProbeDifference(data))
	{
		results = specialised_solver1_->Solve(data);
	}
	else
	{
		results = specialised_solver2_->Solve(data);
	}

	// It might be that a node with a lower depth or fewer nodes is better than one that recurses deeper,
	//   due to the at-least-one-event-per-leaf constraint. Therefore we need to check whether the nodes
	//   can be better when using less nodes
	if (results.node_budget_one.error < results.node_budget_two.error) {
		results.node_budget_two = results.node_budget_one;
	}
	if (results.node_budget_two.error < results.node_budget_three.error) {
		results.node_budget_three = results.node_budget_two;
	}

	stats_.time_in_terminal_node += double(clock() - clock_start) / CLOCKS_PER_SEC;

	std::vector<InternalNodeDescription> optimised_roots;
	optimised_roots.push_back(results.node_budget_one);
	optimised_roots.push_back(results.node_budget_two);
	optimised_roots.push_back(results.node_budget_three);

	//since the specialised algorithm computes trees of size 1, 2, 3 for depth 2
	//	we store all these results in the cache to avoid possibly recomputing later

	if (!cache_->IsOptimalAssignmentCached(data, branch, 1, 1))
	{
		InternalNodeDescription& root_node = results.node_budget_one;
		runtime_assert(results.node_budget_one.NumNodes() <= 1);
		cache_->StoreOptimalBranchAssignment(data, branch, root_node, 1, 1);
	}

	if (!cache_->IsOptimalAssignmentCached(data, branch, 2, 2))
	{
		InternalNodeDescription& root_node = results.node_budget_two;
		runtime_assert(results.node_budget_two.NumNodes() <= 2);
		cache_->StoreOptimalBranchAssignment(data, branch, root_node, 2, 2);
	}

	if (!cache_->IsOptimalAssignmentCached(data, branch, 2, 3))
	{
		InternalNodeDescription& root_node = results.node_budget_three;
		runtime_assert(results.node_budget_three.NumNodes() <= 3);
		cache_->StoreOptimalBranchAssignment(data, branch, root_node, 2, 3);
	}

	//recall that trees with k may be as good as trees with k' nodes, where k' < k
	//we prefer smaller trees
	//find the best smallest tree to return
	InternalNodeDescription best_root_node = optimised_roots[0];//one node
	runtime_assert(best_root_node.NumNodes() <= num_nodes);
	if (num_nodes >= 2 && IsNodeBetter(optimised_roots[1], best_root_node)) { best_root_node = optimised_roots[1]; }
	if (num_nodes == 3 && IsNodeBetter(optimised_roots[2], best_root_node)) { best_root_node = optimised_roots[2]; }

	return (best_root_node.Error() >= upper_bound ? CreateInfeasibleNodeDescription() : best_root_node);
}

InternalNodeDescription Solver::CreateInfeasibleNodeDescription()
{
	return InternalNodeDescription::CreateInfeasibleNodeDescription();
}

InternalNodeDescription Solver::CreateLeafNodeDescription(BinaryDataInternal& data)
{
	int feature = INT32_MAX;
	double theta = LeafTheta(data);
	double error = LeafError(data);
	return InternalNodeDescription(feature, theta, error, 0, 0);
}

InternalNodeDescription Solver::CombineLeftAndRightChildren(int feature, InternalNodeDescription& left_child, InternalNodeDescription& right_child)
{
	double theta = DBL_MAX;
	double error = left_child.error + right_child.error;
	int num_nodes_left = left_child.NumNodes();
	int num_nodes_right = right_child.NumNodes();
	return InternalNodeDescription(feature, theta, error, num_nodes_left, num_nodes_right);
}

ChildSubtreeOrdering Solver::GetSortedChildren(ChildSubtreeInfo& left_child, ChildSubtreeInfo& right_child)
{
	double leaf_left = LeafError(*left_child.binary_data);
	double leaf_right = LeafError(*right_child.binary_data);

	//int lb_left = cache_->RetrieveLowerBound(*left_child.binary_data, left_child.branch, left_child.depth, left_child.size);
	//int lb_right = cache_->RetrieveLowerBound(*right_child.binary_data, right_child.branch, right_child.depth, right_child.size);

	bool go_left_first = (!dynamic_child_selection_) || (leaf_left > leaf_right);
	//go_left_first = (leaf_left - lb_left > leaf_right - lb_right);

	if (go_left_first) { return ChildSubtreeOrdering(left_child, right_child); }
	else { return ChildSubtreeOrdering(right_child, left_child); }
}

double Solver::LeafTheta(BinaryDataInternal& data)
{
	int event_sum = 0;
	double hazard_sum = 0;

	for (FeatureVectorBinary* fv : data.GetInstances())
	{
		event_sum += fv->GetEvent();
		hazard_sum += fv->GetHazard();
	}

	runtime_assert(event_sum > 1e-9);

	return event_sum / hazard_sum;
}

double Solver::LeafError(BinaryDataInternal& data)
{
	double error = 0;
	double theta = LeafTheta(data);

	for (FeatureVectorBinary* fv : data.GetInstances())
	{
		double hazard = fv->GetHazard();

		error += theta * hazard;
		if (fv->GetEvent()) {
			error -= log(hazard) + log(theta) + 1;
		}
	}

	return std::max(error, 0.0);
}

bool Solver::IsTerminalNode(int depth, int max_num_nodes)
{
	return depth <= 2;
}

bool Solver::IsNodeBetter(InternalNodeDescription node1, InternalNodeDescription node2)
{
	runtime_assert(node1.IsFeasible() && node2.IsFeasible());

	if (node1.Error() < node2.Error()) { return true; }
	else if (node1.Error() > node2.Error()) { return false; }
	return node1.NumNodes() < node2.NumNodes();
}

}