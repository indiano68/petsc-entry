#!/bin/bash
working_dir="$(pwd)"
cd $PETSC_DIR
source ./update_headers.sh
make libs hpddmbuild
return_status="$?"

if ! [ "$return_status" == "0" ]; then 
  exit $return_status
fi
cd $working_dir

test_dir="/Users/erikfabrizzi/Workspace/LIP6-SWE/repos/petsc-entry"
make -f ${PETSC_DIR}/gmakefile.test TESTSRCDIR=$test_dir TESTDIR=$test_dir/tests-build pkgs=petsc-entry

test_dir="/Users/erikfabrizzi/Workspace/LIP6-SWE/repos/petsc-entry/tests"

cwd="$(pwd)"
cwd="$(realpath $cwd)"
cd $test_dir 

if [ -e 'noiseless_cpu' ]; then
  rm noiseless_cpu
fi
if [ -e 'noiseless_cpu_optim' ]; then
  rm noiseless_cpu
fi
if [ -e 'noiseless_cpu_debug' ]; then
  rm noiseless_cpu
fi

export PETSC_ARCH=arch-debug
make noiseless_cpu
mv noiseless_cpu noiseless_cpu_debug

export PETSC_ARCH=arch-opt
make noiseless_cpu
mv noiseless_cpu noiseless_cpu_opt

cd $cwd
