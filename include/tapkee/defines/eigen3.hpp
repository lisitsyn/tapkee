#pragma once

//// Eigen 5 library includes
#ifdef TAPKEE_EIGEN_INCLUDE_FILE
#include TAPKEE_EIGEN_INCLUDE_FILE
#else
#ifndef TAPKEE_DEBUG
#define EIGEN_NO_DEBUG
#endif
#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#if defined(TAPKEE_SUPERLU_AVAILABLE) && defined(TAPKEE_USE_SUPERLU)
#include <Eigen/SuperLUSupport>
#endif
#endif

#ifdef EIGEN_RUNTIME_NO_MALLOC
#define RESTRICT_ALLOC Eigen::internal::set_is_malloc_allowed(false)
#define UNRESTRICT_ALLOC Eigen::internal::set_is_malloc_allowed(true)
#else
#define RESTRICT_ALLOC
#define UNRESTRICT_ALLOC
#endif
//// end of Eigen library includes
