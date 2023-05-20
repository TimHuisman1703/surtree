//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#include "internal_node_description.h"
#include "../Utilities/runtime_assert.h"

namespace MurTree
{
InternalNodeDescription::InternalNodeDescription(int feature, double theta, double error, int num_nodes_left, int num_nodes_right):
    feature(feature),
    theta(theta),
    error(error),
    num_nodes_left(num_nodes_left),
    num_nodes_right(num_nodes_right)
{
}

double InternalNodeDescription::Error() const
{
    runtime_assert(IsFeasible());
    return error;
}

double InternalNodeDescription::SparseObjective(double sparse_coefficient) const
{
    return error + sparse_coefficient * NumNodes();
}

int InternalNodeDescription::NumNodes() const
{
    if (feature == INT32_MAX) { return 0; }
    else { return 1 + num_nodes_left + num_nodes_right; }
}

bool InternalNodeDescription::IsInfeasible() const
{
    return !IsFeasible();
}

bool InternalNodeDescription::IsFeasible() const 
{
    return error != DBL_MAX;
}

InternalNodeDescription InternalNodeDescription::CreateInfeasibleNodeDescription()
{
    return InternalNodeDescription(INT32_MAX, DBL_MAX, DBL_MAX, INT32_MAX, INT32_MAX);
}

}//end namespace MurTree