#!/bin/bash

set -Eeuo pipefail
trap 'echo "Error on line $LINENO"; exit 1' ERR

ROOT_DIR="$(pwd)"

if [ -e ".envfile" ]; then
    source .envfile
fi

PETSC_DIR=${PETSC_DIR:-""}
if [ -z "$PETSC_DIR" ]; then
    echo "ERROR: PETSC_DIR unset. Stopping..."
    exit 1
fi

PETSC_ARCH=${PETSC_ARCH:-""}
if [ -z "$PETSC_ARCH" ]; then
    echo "ERROR: PETSC_ARCH unset. Stopping..."
    exit 1
fi

SLEPC_DIR=${SLEPC_DIR:-"$PETSC_DIR/$PETSC_ARCH"}
if [ ! -f "$SLEPC_DIR/lib/slepc/conf/slepc_common" ]; then
    echo "ERROR: SLEPC not found at $SLEPC_DIR. Stopping..."
    exit 1
fi

BIN_DIR="$ROOT_DIR/bin/$PETSC_ARCH"
mkdir -p "$BIN_DIR"

cd "$ROOT_DIR/slepc"
if [ -e "minimaleps" ]; then
    rm "minimaleps"
fi
make -f makefile.minimaleps minimaleps
mv "minimaleps" "$BIN_DIR/minimaleps"

cd "$ROOT_DIR"
echo "Success!"
