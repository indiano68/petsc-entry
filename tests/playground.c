#include "petscmat.h"
#include <petsc.h>
#include <petscsys.h>

int main(int argc, char **argv) {
  Mat A;    /* 5x5 on each rank */
  Vec d;    /* global distributed (MPICUDA) vector: 5 entries per rank */
  Vec dloc; /* local (COMM_SELF) sequential vector (created by us) */
  PetscRandom rctx;
  PetscMPIInt rank, size;

  PetscInitialize(&argc, &argv, NULL, NULL);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

/* Local matrix on each rank */
#if PetscDefined(HAVE_CUDA)
  MatCreateSeqDenseCUDA(PETSC_COMM_SELF, 5, 5, NULL, &A);
#else
  MatCreateSeqDense(PETSC_COMM_SELF, 5, 5, NULL, &A);
#endif

  /* Global distributed vector: 5 entries per rank */
  VecCreateMPI(PETSC_COMM_WORLD, 5, PETSC_DECIDE, &d);
  VecSetType(d, VECMPICUDA);

  /* Random matrix */
  PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
  PetscRandomSetFromOptions(rctx);

  MatSetRandom(A, rctx);
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  /* Fill global vector with alternating 1/0 using global indices */
  {
    PetscInt rstart, rend, i;
    VecGetOwnershipRange(d, &rstart, &rend); /* expect rend-rstart == 5 */

    for (i = rstart; i < rend; i++) {
      PetscScalar v = ((i % 2) == 0) ? 1.0 : 0.0;
      VecSetValue(d, i, v, INSERT_VALUES);
    }
    VecAssemblyBegin(d);
    VecAssemblyEnd(d);
  }

  /* Optional prints */
  PetscPrintf(PETSC_COMM_WORLD, "\n=== Rank %d/%d ===\n", rank, size);
  PetscPrintf(PETSC_COMM_WORLD, "A before scaling (rank %d):\n", rank);
  MatView(A, PETSC_VIEWER_STDOUT_WORLD);

  /* Create a local sequential Vec and use VecGetLocalVector/RestoreLocalVector
   * (by value) */
  {
    PetscInt nloc;
    VecGetLocalSize(d, &nloc); /* should be 5 */
    VecCreateSeq(PETSC_COMM_SELF, nloc, &dloc);

    VecGetLocalVector(d, dloc); /* NOTE: second arg is Vec, not Vec* */
    MatDiagonalScale(A, dloc, NULL);
    VecRestoreLocalVector(d, dloc);

    VecDestroy(&dloc);
  }

  PetscPrintf(PETSC_COMM_WORLD, "A after scaling (rank %d):\n", rank);
  MatView(A, PETSC_VIEWER_STDOUT_WORLD);

  PetscRandomDestroy(&rctx);
  VecDestroy(&d);
  MatDestroy(&A);

  PetscFinalize();
  return 0;
}
