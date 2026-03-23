#include "petscksp.h"
#include "petscmat.h"
#include "petscpc.h"
#include "petscpctypes.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <petsc.h>
#include <petscsys.h>

int main(int argc, char **argv) {
  Mat A;
  Vec b, b_cp, x;
  KSP ksp;
  PC pc;
  const PetscInt m = 100, n = 100;

  PetscInitialize(&argc, &argv, NULL, NULL);
  MatCreate(PETSC_COMM_WORLD, &A);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n);
  MatSetRandom(A, NULL);
  MatCreateVecs(A, &b, &x);
  VecSetRandom(b, NULL);
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetOperators(ksp, A, A);
  KSPSetType(ksp, KSPPREONLY);
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCLU);
  KSPSolve(ksp, b, x);
  KSPDestroy(&ksp);
  MatDestroy(&A);
  VecDestroy(&b);
  VecDestroy(&x);
  PetscFinalize();
}
