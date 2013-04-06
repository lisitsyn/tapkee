In this example the simplest case of using the Tapkee library is considered. For
the sake of simplicity, the data used in this example is one dimensional. Therefore,
it actually does not make sense do apply dimension reduction since the dimension
of the input data cannot be reduced. The method used is Multidimensional Scaling,
or MDS for short, and the main callback used is called my_distance_callback, which
simply computes the absolute value of the difference between two data points.
Note that there also appear two other callbacks, a kernel callback and feature
vector callback. However, these will not be used since the macro
TAPKEE_CALLBACK_IS_DISTANCE tells Tapkee that only a distance callback is required
in this example. The kernel and feature vector callbacks only appear due to API
consistency.
