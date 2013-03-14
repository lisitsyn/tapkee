#!/usr/bin/env bash

echo "===== Locally Linear Embedding (k=15) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m lle -k 15
echo "===== Neighborhood preserving embedding (k=15) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m npe -k 15
echo "===== Local Tangent Space Alignment (k=15) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m ltsa -k 15
echo "===== Linear Local Tangent Space Alignment (k=15) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m lltsa -k 15
echo "===== Hessian Locally Linear Embedding (k=15) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m hlle -k 15
echo "===== Diffusion map (t=3, w=1000.0, arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m diffusion_map -t 3 -w 10.0
echo "===== Diffusion map (t=3, w=1000.0, randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m diffusion_map -em randomized -t 3 -w 10.0
echo "===== Multidimensional scaling (arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m mds
echo "===== Multidimensional scaling (randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m mds -em randomized
echo "===== Landmark Multidimensional scaling (arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m l-mds
echo "===== Landmark Multidimensional scaling (randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m l-mds -em randomized
echo "===== Isomap (k=15, arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m isomap -k 15
echo "===== Isomap (k=15, randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m isomap -k 15 -em randomized
echo "===== Landmark Isomap (k=15, arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m l-isomap -k 15 --landmark-ratio 0.1
echo "===== Landmark Isomap (k=15, randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m l-isomap -k 15 -em randomized --landmark-ratio 0.1
echo "===== Laplacian Eigenmaps (w=1000.0) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m laplacian_eigenmaps -w 1000.0
echo "===== Locality Preserving Projections (w=1000.0) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m lpp -w 1000.0
echo "===== PCA (arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m pca 
echo "===== PCA (randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m pca -em randomized
echo "===== Kernel PCA (arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m kpca
echo "===== Kernel PCA (randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m kpca -em randomized

