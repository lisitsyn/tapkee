#pragma once
// Eigen is already included by RcppEigen.h.
// This file is used as TAPKEE_EIGEN_INCLUDE_FILE so that tapkee's
// eigen3.hpp skips its own Eigen include and EIGEN_RUNTIME_NO_MALLOC.
// Since EIGEN_RUNTIME_NO_MALLOC is not defined, RESTRICT_ALLOC and
// UNRESTRICT_ALLOC are automatically defined as no-ops by eigen3.hpp.
