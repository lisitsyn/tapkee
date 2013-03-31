#ifndef TAPKEE_MATRIX_H_
#define TAPKEE_MATRIX_H_

namespace tapkee 
{
namespace tapkee_internal
{

template <class Callback>
class CallbackMatrix : public tapkee::DenseMatrix
{
public:

	typedef tapkee::DenseMatrix Base;
}

} // End of tapkee_internal
} // End of tapkee
#endif

