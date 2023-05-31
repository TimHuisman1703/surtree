//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#pragma once

#include "../Data Structures/binary_data.h"

namespace MurTree
{
struct SplitBinaryData
{
	SplitBinaryData(int num_features):
		data_without_feature(num_features),
		data_with_feature(num_features),
		events_without_feature(0),
		events_with_feature(0)
	{
	}


	SplitBinaryData(int splitting_feature, BinaryDataInternal &data):
		data_without_feature(data.NumFeatures()),
		data_with_feature(data.NumFeatures()),
		events_without_feature(0),
		events_with_feature(0)
	{
		SplitData(splitting_feature, data);
	}

	void SplitData(int splitting_feature, BinaryDataInternal& data)
	{
		data_without_feature.Clear();
		data_with_feature.Clear();
		events_without_feature = 0;
		events_with_feature = 0;
		
		for (FeatureVectorBinary *fv : data.GetInstances())
		{
			if (fv->IsFeaturePresent(splitting_feature))
			{
				data_with_feature.AddFeatureVector(fv);
				events_with_feature += fv->GetEvent();
			}
			else
			{
				data_without_feature.AddFeatureVector(fv);
				events_without_feature += fv->GetEvent();
			}
		}
	}

	BinaryDataInternal data_without_feature, data_with_feature;
	int events_without_feature, events_with_feature;
};
}