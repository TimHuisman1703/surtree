//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#pragma once

#define AVERAGE_LOGLIKELIHOOD

#include "../Utilities/runtime_assert.h"
#include "binary_data.h"

namespace MurTree
{
//TODO Memory leaks
struct DecisionNode
{
	static DecisionNode* CreateLabelNode(double theta)
	{
		DecisionNode* node = new DecisionNode();
		node->feature_ = INT32_MAX;
		node->theta_ = theta;
		node->left_child_ = NULL;
		node->right_child_ = NULL;
		return node;
	}
	static DecisionNode* CreateFeatureNodeWithNullChildren(int feature)
	{
		DecisionNode* node = new DecisionNode();
		node->feature_ = feature;
		node->theta_ = DBL_MAX;
		node->left_child_ = NULL;
		node->right_child_ = NULL;
		return node;
	}
	
	int Depth() const
	{
		if (IsLabelNode()) { return 0; }
		return 1 + std::max(left_child_->Depth(), right_child_->Depth());		
	}

	int NumNodes() const
	{
		if (IsLabelNode()) { return 0; }
		return 1 + left_child_->NumNodes() + right_child_->NumNodes();
	}

	bool IsLabelNode() const { return feature_ == INT32_MAX; }
	bool IsFeatureNode() const { return feature_ != INT32_MAX; }

	double ComputeError(BinaryDataInternal& data)
	{
		double error = 0.0;
		for (FeatureVectorBinary* fv : data.GetInstances())
		{
			double theta = Classify(fv);
			double hazard = fv->GetHazard();

			error += theta * hazard;
			if (fv->GetEvent()) {
				error -= log(hazard) + log(theta) + 1;
			}
		}

#ifdef AVERAGE_LOGLIKELIHOOD
		error = error / data.GetInstances().size();
#endif

		return std::max(error, 0.0);
	}

	double ComputeError(std::vector<FeatureVectorBinary> instances)
	{
		double error = 0.0;
		for (FeatureVectorBinary fv : instances)
		{
			double theta = Classify(&fv);
			double hazard = fv.GetHazard();

			error += theta * hazard;
			if (fv.GetEvent()) {
				error -= log(hazard) + log(theta) + 1;
			}
		}

#ifdef AVERAGE_LOGLIKELIHOOD
		error = error / instances.size();
#endif

		return std::max(error, 0.0);
	}
	
	double Classify(FeatureVectorBinary* feature_vector)
	{
		if (IsLabelNode())
		{
			return theta_;
		}
		else if (feature_vector->IsFeaturePresent(feature_))
		{
			return right_child_->Classify(feature_vector);
		}
		else
		{
			return left_child_->Classify(feature_vector);
		}
	}

	int feature_;
	double theta_;
	DecisionNode* left_child_, * right_child_;
};
}//end namespace MurTree