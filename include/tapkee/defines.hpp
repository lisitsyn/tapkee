/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_DEFINES_H_
#define TAPKEE_DEFINES_H_

/* Tapkee includes */
#include <tapkee/exceptions.hpp>
#include <tapkee/parameters/parameters.hpp>
#include <tapkee/traits/callbacks_traits.hpp>
#include <tapkee/traits/methods_traits.hpp>
/* End of Tapkee includes */

#include <string>

#define TAPKEE_WORLD_VERSION 1
#define TAPKEE_MAJOR_VERSION 0
#define TAPKEE_MINOR_VERSION 0

/* Tapkee includes */
#include <tapkee/defines/eigen3.hpp>
#include <tapkee/defines/types.hpp>
#include <tapkee/defines/methods.hpp>
#include <tapkee/defines/keywords.hpp>
#include <tapkee/defines/stdtypes.hpp>
#include <tapkee/defines/synonyms.hpp>
#include <tapkee/defines/random.hpp>
#include <tapkee/projection.hpp>
/* End of Tapkee includes */

#ifdef TAPKEE_CUSTOM_PROPERTIES
	#include TAPKEE_CUSTOM_PROPERTIES
#else
	//! Base of covertree. Could be overrided if TAPKEE_CUSTOM_PROPERTIES file is defined.
	#define COVERTREE_BASE 1.3
#endif

namespace tapkee
{
	//! Return result of the library - a pair of @ref DenseMatrix (embedding) and @ref ProjectingFunction
	struct TapkeeOutput
	{
		TapkeeOutput() :
			embedding(), projection()
		{
		}
		TapkeeOutput(const tapkee::DenseMatrix& e, const tapkee::ProjectingFunction& p) :
			embedding(), projection(p)
		{
			embedding.swap(e);
		}
		TapkeeOutput(const TapkeeOutput& that) :
			embedding(), projection(that.projection)
		{
			this->embedding.swap(that.embedding);
		}
		tapkee::DenseMatrix embedding;
		tapkee::ProjectingFunction projection;
	};
}

#endif // TAPKEE_DEFINES_H_
