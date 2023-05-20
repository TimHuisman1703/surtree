//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#include "binary_data.h"
#include "../Utilities/runtime_assert.h"

#include <numeric>

namespace MurTree
{
BinaryDataInternal::BinaryDataInternal(int num_features):
	num_features_(num_features),
	hash_value_(-1),
	is_closure_set_(false)
{
}

const FeatureVectorBinary* BinaryDataInternal::GetInstance(int index) const
{
	return instances_[index];
}

const std::vector<FeatureVectorBinary*>& BinaryDataInternal::GetInstances() const
{
	return instances_;
}

FeatureVectorBinary* BinaryDataInternal::GetInstance(int index)
{
	return instances_[index];
}

int BinaryDataInternal::NumFeatures() const
{
	return num_features_;
}

int BinaryDataInternal::NumInstances() const
{
	return (int)instances_.size();
}

int BinaryDataInternal::MaxFeatureVectorID() const
{
	int max_id = 0;
	for (const FeatureVectorBinary* fv : instances_)
	{
		max_id = std::max(max_id, int(fv->GetID()));
	}
	return max_id + 1;
}

void BinaryDataInternal::AddFeatureVector(FeatureVectorBinary* fv)
{
	assert(num_features_ == fv->NumTotalFeatures());
	instances_.push_back(fv);
}

int BinaryDataInternal::Size() const
{
	return int(instances_.size());
}

bool BinaryDataInternal::IsEmpty() const
{
	return Size() == 0;
}

void BinaryDataInternal::Clear()
{
	instances_.clear();
	hash_value_ = -1;
	is_closure_set_ = false;
}

double BinaryDataInternal::ComputeSparsity() const
{
	if (Size() == 0) { return 0.0; }
	double sum_sparsity = 0.0;
	for (const FeatureVectorBinary* fv : instances_)
	{
		sum_sparsity += fv->Sparsity();
	}
	return sum_sparsity / (Size());
}

void BinaryDataInternal::PrintStats()
{
	std::cout << "Num features: " << NumFeatures() << "\n";
	std::cout << "Size: " << Size() << "\n";
	std::cout << "Sparsity: " << ComputeSparsity() << "\n";
}

bool BinaryDataInternal::IsHashSet() const
{
	return hash_value_ != -1;
}

int BinaryDataInternal::GetHash() const
{
	runtime_assert(hash_value_ != -1); 
	return (int)hash_value_;
}

void BinaryDataInternal::SetHash(int new_hash)
{
	runtime_assert(new_hash != -1);
	hash_value_ = new_hash;
}

bool BinaryDataInternal::IsClosureSet() const
{
	return is_closure_set_;
}

Branch &BinaryDataInternal::GetClosure()
{
	runtime_assert(IsClosureSet());
	return closure_;
}

void BinaryDataInternal::SetClosure(const Branch& closure)
{
	closure_ = closure;
	is_closure_set_ = true;
}

std::ostream& operator<<(std::ostream& os, BinaryDataInternal& data)
{
	os << "Number of instances: " << data.Size();

	int counter_instances = 0;
	for (FeatureVectorBinary* fv : data.GetInstances())
	{
		os << "\n\t" << counter_instances++ << ":\t" << (*fv);
	}
	return os;
}
}
