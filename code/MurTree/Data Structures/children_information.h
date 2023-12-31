﻿//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#pragma once

#include <stdint.h>

struct ChildrenInformation
{
	/*ChildrenInformation() {
		left_child_feature = right_child_feature = left_child_penalty = right_child_penalty = DBL_MAX;
	}*/

	/*ChildrenInformation() { Initialise(); }

	void Initialise()
	{
		left_child_feature = right_child_feature = left_child_penalty = right_child_penalty = DBL_MAX;
	}

	void UpdateBestLeftChild(int left_child_feature, int penalty)
	{
		if (left_child_penalty > penalty)
		{
			left_child_feature = left_child_feature;
			left_child_penalty = penalty;
		}
	}

	void UpdateBestRightChild(int right_child_feature, int penalty)
	{
		if (right_child_penalty > penalty)
		{
			right_child_feature = right_child_feature;
			right_child_penalty = penalty;
		}
	}*/

	int left_child_feature, right_child_feature;
	double left_child_penalty, right_child_penalty;
};