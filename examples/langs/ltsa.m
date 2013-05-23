N = 1000;
tt = 5*pi/4*(1+2*rand(1,N));
height = rand(1,N)-0.5;
noise = 0.01;
matrix = [tt+noise.*rand(1,N).*cos(tt);
          10*height;
          tt+noise.*rand(1,N).*sin(tt)];

init_shogun

# load data
# create features instance
features = RealFeatures(matrix);

# create Local Tangent Space Alignment converter instance
converter = LocalTangentSpaceAlignment();

# set target dimensionality
converter.set_target_dim(2);
# set number of neighbors 
converter.set_k(20);
# set nullspace shift (optional)
converter.set_nullspace_shift(-1e-6);

# compute embedding with Local Tangent Space Alignment method
embedding = converter.embed(features);
