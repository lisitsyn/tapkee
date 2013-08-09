#!/usr/bin/env bash

echo "===== Locally Linear Embedding (k=15) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m lle -k 15 | $TRANSFORM
echo "===== Neighborhood preserving embedding (k=15) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m npe -k 15 | $TRANSFORM
echo "===== Local Tangent Space Alignment (k=15) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m ltsa -k 15 | $TRANSFORM
echo "===== Linear Local Tangent Space Alignment (k=15) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m lltsa -k 15 | $TRANSFORM
echo "===== Hessian Locally Linear Embedding (k=15) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m hlle -k 15 | $TRANSFORM
echo "===== Diffusion map (t=2, w=10.0, arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m diffusion_map -t 2 -gw 10.0 | $TRANSFORM
echo "===== Diffusion map (t=2, w=10.0, randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m diffusion_map -em randomized -t 2 -gw 10.0 | $TRANSFORM
echo "===== Multidimensional scaling (arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m mds | $TRANSFORM
echo "===== Multidimensional scaling (randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m mds -em randomized | $TRANSFORM
echo "===== Landmark Multidimensional scaling (arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m l-mds | $TRANSFORM
echo "===== Landmark Multidimensional scaling (randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m l-mds -em randomized | $TRANSFORM
echo "===== Isomap (k=15, arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m isomap -k 15 | $TRANSFORM
echo "===== Isomap (k=15, randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m isomap -k 15 -em randomized | $TRANSFORM
echo "===== Landmark Isomap (k=15, arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m l-isomap -k 15 --landmark-ratio 0.1 | $TRANSFORM
echo "===== Landmark Isomap (k=15, randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m l-isomap -k 15 -em randomized --landmark-ratio 0.1 | $TRANSFORM
echo "===== Laplacian Eigenmaps (w=1000.0) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m laplacian_eigenmaps -gw 1000.0 | $TRANSFORM
echo "===== Locality Preserving Projections (w=1000.0) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m lpp -gw 1000.0 | $TRANSFORM
echo "===== PCA (arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m pca  | $TRANSFORM
echo "===== PCA (randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m pca -em randomized | $TRANSFORM
echo "===== Kernel PCA (arpack) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m kpca | $TRANSFORM
echo "===== Kernel PCA (randomized) ======"
$CALLENV $TAPKEE_ELF -i $INPUT_FILE -o $OUTPUT_FILE -m kpca -em randomized | $TRANSFORM

