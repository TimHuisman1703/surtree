//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#include "feature_vector_binary.h"
#include "../Utilities/runtime_assert.h"

#include <iostream>
#include <assert.h>

namespace MurTree
{
FeatureVectorBinary::FeatureVectorBinary() :
	id_(-1),
	is_feature_present_(NULL),
	time_(0),
	event_(0),
	hazard_(-1)
{ }

FeatureVectorBinary::FeatureVectorBinary(const std::vector<bool>& feature_values, double time, int event, int id):
	id_(id),
	is_feature_present_(feature_values),
	time_(time),
	event_(event),
	hazard_(-1)
{
	for (size_t feature_index = 0; feature_index < feature_values.size(); feature_index++)
	{
		if (feature_values[feature_index] == true) { present_features_.push_back((int)feature_index); }
	}
}

bool FeatureVectorBinary::IsFeaturePresent(int feature) const
{
	return is_feature_present_[feature];
}

int FeatureVectorBinary::GetJthPresentFeature(int j) const
{
	return present_features_[j];
}

double FeatureVectorBinary::Sparsity() const
{
	return double(NumPresentFeatures()) / is_feature_present_.size();
}

int FeatureVectorBinary::NumPresentFeatures() const
{
	return int(present_features_.size());
}

int FeatureVectorBinary::NumTotalFeatures() const
{
	return (int)is_feature_present_.size();
}

double FeatureVectorBinary::GetTime() const
{
	return time_;
}

int FeatureVectorBinary::GetEvent() const
{
	return event_;
}

double FeatureVectorBinary::GetHazard() const
{
	runtime_assert(hazard_ > -0.5)
	return hazard_;
}

int FeatureVectorBinary::GetID() const
{
	return id_;
}

double FeatureVectorBinary::SetHazard(double new_value)
{
	bool old = hazard_;
	hazard_ = new_value;
	return old;
}

typename std::vector<int>::const_iterator FeatureVectorBinary::begin() const
{
	return present_features_.begin();
}

typename std::vector<int>::const_iterator FeatureVectorBinary::end() const
{
	return present_features_.end();
}

std::ostream& operator<<(std::ostream& os, const FeatureVectorBinary& fv)
{
	if (fv.NumPresentFeatures() == 0) { std::cout << "[empty]"; }
	else
	{
		auto iter = fv.begin();
		std::cout << *iter;
		++iter;
		while (iter != fv.end())
		{
			std::cout << " " << *iter;
			++iter;
		}
	}
	return os;
}

}//end namespace MurTree