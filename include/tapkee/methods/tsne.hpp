/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/external/barnes_hut_sne/tsne.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(tDistributedStochasticNeighborEmbedding)
    TapkeeOutput embed()
    {
        parameters[sne_perplexity].checked().satisfies(InClosedRange<ScalarType>(0.0, (n_vectors - 1) / 3.0)).orThrow();
        parameters[sne_theta].checked().satisfies(NonNegativity<ScalarType>()).orThrow();

        DenseMatrix data = dense_matrix_from_features(features, current_dimension, begin, end);

        DenseMatrix embedding(static_cast<IndexType>(parameters[target_dimension]), n_vectors);
        tsne::TSNE tsne;
        tsne.run(data, data.cols(), data.rows(), embedding.data(), parameters[target_dimension],
                 parameters[sne_perplexity], parameters[sne_theta]);

        return TapkeeOutput(embedding.transpose(), unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
