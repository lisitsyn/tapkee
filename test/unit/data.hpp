#include <cmath>

#include <fstream>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

tapkee::DenseMatrix swissroll(int N) {
	tapkee::DenseVector tt = (3.0*M_PI/4.0)*(tapkee::DenseVector::Random(N).array()+0.5);
	tapkee::DenseVector height = tapkee::DenseVector::Random(N).array() - 0.5;
	tapkee::DenseMatrix X(N,3);
	X.col(0) = tt.array()*tt.array().cos();
	X.col(1) = 10*height;
	X.col(2) = tt.array()*tt.array().sin();
	X.transposeInPlace();
	return X;
}

