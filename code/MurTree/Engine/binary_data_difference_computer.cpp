﻿//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#include "binary_data_difference_computer.h"

namespace MurTree
{
DifferenceMetrics BinaryDataDifferenceComputer::ComputeDifferenceMetrics(BinaryDataInternal& data_old, BinaryDataInternal& data_new)
{
	DifferenceMetrics metrics;

	int size_new = data_new.NumInstances();
	int size_old = data_old.NumInstances();
	int index_new = 0, index_old = 0;
	while (index_new < size_new && index_old < size_old)
	{
		int id_new = data_new.GetInstance(index_new)->GetID();
		int id_old = data_old.GetInstance(index_old)->GetID();

		//the new data has something the old one does not
		if (id_new < id_old)
		{
			metrics.total_difference++;
			index_new++;
		}
		//the old data has something the new one does not
		else if (id_new > id_old)
		{
			metrics.total_difference++;
			metrics.num_removals++;
			index_old++;
		}
		else
		{//no difference
			index_new++;
			index_old++;
		}
			
			
	}
	int num_new_instances = (size_new - index_new);
	int num_removed_instances = (size_old - index_old);
	metrics.total_difference += num_new_instances;
	metrics.total_difference += num_removed_instances;
	metrics.num_removals += num_removed_instances;
	
	return metrics;
}

void BinaryDataDifferenceComputer::ComputeDifference(BinaryDataInternal& data_old, BinaryDataInternal& data_new, BinaryDataInternal& data_to_add, BinaryDataInternal& data_to_remove)
{
	data_to_add.Clear();
	data_to_remove.Clear();

	int size_new = data_new.NumInstances();
	int size_old = data_old.NumInstances();
	int index_new = 0, index_old = 0;
	while (index_new < size_new && index_old < size_old)
	{
		FeatureVectorBinary *fv_new = data_new.GetInstance(index_new);
		FeatureVectorBinary *fv_old = data_old.GetInstance(index_old);

		//the new data has something the old one does not
		if (fv_new->GetID() < fv_old->GetID()) 
		{
			data_to_add.AddFeatureVector(fv_new);
			index_new++;
		}
		//the old data has something the new one does not
		else if (fv_new->GetID() > fv_old->GetID()) 
		{ 
			data_to_remove.AddFeatureVector(fv_old);
			index_old++;
		}
		else
		{//no difference
			index_new++;
			index_old++;
		}
	}//end while

	while (index_new < size_new)
	{
		FeatureVectorBinary* fv_new = data_new.GetInstance(index_new);
		data_to_add.AddFeatureVector(fv_new);
		index_new++;
	}
		
	while (index_old < size_old)
	{
		FeatureVectorBinary* fv_old = data_old.GetInstance(index_old);
		data_to_remove.AddFeatureVector(fv_old);
		index_old++;
	}
}

}
