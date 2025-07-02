#!/bin/bash

# Exit if any command fails
set -e

# Check for 2 arguments (ntls and nphot)
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_all.sh <ntls> <nphot>"
    exit 1
fi

NTLS=$1
NPHOT=$2

# Debug: Check if the arguments are being passed correctly
echo "Received ntls = $NTLS, nphot = $NPHOT"

# Calculate rho 
echo "Received ntls = $NTLS, and nphot = $NPHOT"
python3 generate_rho.py "$NTLS" "$NPHOT"

# Step-by-step execution
# echo "Running full eigenspace calculation and generate rho steady state"
# python3 eigenv_run_spin_matrix_elements_direct.py "$NTLS" "$NPHOT"

echo "Running full eigenspace calculation and generate rho steady state"
python3 run_spin_matrix_elements_direct_pt.py "$NTLS" "$NPHOT"

# echo "Run for exact eigenvalues"
# python3 eigenvalue_calc/exact_eigenvalues_pt.py "$NTLS" "$NPHOT"

# echo "Run for symmetric eigenvalues"
# python3 eigenvalue_calc/symmetric_eigenvalues.py "$NTLS" "$NPHOT"

# echo "Generate plot"
# python3 plots/plots.py "$NTLS" "$NPHOT"