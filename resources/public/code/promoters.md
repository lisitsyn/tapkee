This example illustrates the use of Tapkee as a module of the Shogun Machine Learning
Toolbox and its integration with other parts of the Toolbox. The data used in
this example is formed by mammalian promoter genes (an important region of the
DNA) that are encoded using strings. The embedding algorithm used is Multidimensional
Scaling (MDS), thus the only callback that Tapkee needs is a distance callback. In
particular, the distance callback used is a string kernel implemented in Shogun
called the Weighted Degree (WD) kernel; this kernel is very common in genomics.
As shown in the figure, the data points are embedded in the Euclidean
plane with promoter genes that are similar to each other represented by points
that are close in the plane.
