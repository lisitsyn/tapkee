/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/methods/kernel_locally_linear_embedding.hpp>
#include <tapkee/methods/kernel_local_tangent_space_alignment.hpp>
#include <tapkee/methods/diffusion_map.hpp>
#include <tapkee/methods/multidimensional_scaling.hpp>
#include <tapkee/methods/landmark_multidimensional_scaling.hpp>
#include <tapkee/methods/isomap.hpp>
#include <tapkee/methods/landmark_isomap.hpp>
#include <tapkee/methods/neighborhood_preserving_embedding.hpp>
#include <tapkee/methods/hessian_locally_linear_embedding.hpp>
#include <tapkee/methods/laplacian_eigenmaps.hpp>
#include <tapkee/methods/locality_preserving_projections.hpp>
#include <tapkee/methods/pca.hpp>
#include <tapkee/methods/random_projection.hpp>
#include <tapkee/methods/kernel_pca.hpp>
#include <tapkee/methods/linear_local_tangent_space_alignment.hpp>
#include <tapkee/methods/stochastic_proximity_embedding.hpp>
#include <tapkee/methods/factor_analysis.hpp>
#include <tapkee/methods/manifold_sculpting.hpp>
#include <tapkee/methods/tsne.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(PassThru)
    void validate()
    {
    }

    TapkeeOutput embed()
    {
        DenseMatrix feature_matrix = dense_matrix_from_features(this->features, this->current_dimension, this->begin, this->end);
        return TapkeeOutput(feature_matrix.transpose(), unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
