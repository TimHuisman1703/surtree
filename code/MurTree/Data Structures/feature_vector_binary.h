//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#pragma once

#include <vector>
#include <fstream>

namespace MurTree
{
	class FeatureVectorBinary
	{
	public:
		FeatureVectorBinary();
		FeatureVectorBinary(const std::vector<bool>& feature_values, double last_checkup_time, int event_occured, int id);

		bool IsFeaturePresent(int feature) const;
		int GetJthPresentFeature(int j) const;
		int NumPresentFeatures() const;
		int NumTotalFeatures() const;
		double GetTime() const;
		int GetEvent() const;
		double GetHazard() const;
		int GetID() const;
		double Sparsity() const;

		double SetHazard(double new_value);

		std::vector<int>::const_iterator begin() const;
		std::vector<int>::const_iterator end() const;

		friend std::ostream& operator<<(std::ostream& os, const FeatureVectorBinary& fv);

	private:
		int id_;
		std::vector<bool> is_feature_present_; //[i] indicates if the feature is true or false, i.e., if it is present in present_Features.
		std::vector<int> present_features_;
		double time_;
		int event_;
		double hazard_;

	};
}