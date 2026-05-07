#include "petscsys.h"
#include <HPDDM_debug.hpp>
#include <limits.h>
#include <petsc.h>
#include <slepceps.h>
#include <stdint.h>

static PetscReal hash_to_unit_interval(PetscInt seed, PetscInt matrix_id,
                                       PetscInt row, PetscInt col) {
  uint64_t x = (uint64_t)(PetscInt64)seed;

  x += 0x9e3779b97f4a7c15ULL * (uint64_t)(matrix_id + 1);
  x ^= 0xbf58476d1ce4e5b9ULL * (uint64_t)(row + 1);
  x ^= 0x94d049bb133111ebULL * (uint64_t)(col + 1);

  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;

  return (PetscReal)(x >> 11) / (PetscReal)(1ULL << 53);
}

static PetscScalar deterministic_entry(PetscInt seed, PetscInt matrix_id,
                                       PetscInt row, PetscInt col) {
  PetscReal real_part = hash_to_unit_interval(seed, matrix_id, row, col);
  return real_part;
}

static PetscScalar symmetric_entry(PetscInt seed, PetscInt matrix_id,
                                   PetscInt row, PetscInt col) {
  PetscInt i = row < col ? row : col;
  PetscInt j = row < col ? col : row;

  return deterministic_entry(seed, matrix_id, i, j);
}

static PetscErrorCode FillMatrixDeterministically(Mat M, PetscInt seed,
                                                  PetscInt matrix_id) {
  PetscInt rstart, rend, ncols, i, j;

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(M, &rstart, &rend));
  PetscCall(MatGetSize(M, NULL, &ncols));
  for (i = rstart; i < rend; ++i) {
    for (j = 0; j < ncols; ++j) {
      PetscScalar value;

      if (matrix_id == 0) {
        value = symmetric_entry(seed, matrix_id, i, j);
      } else {
        value = (i == j) ? (PetscScalar)(1.0 + hash_to_unit_interval(seed, matrix_id, i, j))
                         : (PetscScalar)0.0;
      }

      PetscCall(MatSetValue(M, i, j, value, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CanonicalizeEigenvectorSign(Vec v) {
  const PetscScalar *array;
  PetscMPIInt owner, rank;
  PetscInt first = PETSC_MAX_INT, global_first, nlocal, offset;
  PetscScalar pivot = 0.0;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)v), &rank));
  PetscCall(VecGetOwnershipRange(v, &offset, NULL));
  PetscCall(VecGetLocalSize(v, &nlocal));
  PetscCall(VecGetArrayRead(v, &array));
  for (PetscInt i = 0; i < nlocal; ++i) {
    if (PetscAbsScalar(array[i]) > 1.0e-12) {
      first = offset + i;
      break;
    }
  }
  PetscCall(VecRestoreArrayRead(v, &array));
  PetscCallMPI(MPI_Allreduce(&first, &global_first, 1, MPIU_INT, MPI_MIN,
                             PetscObjectComm((PetscObject)v)));
  if (global_first == PETSC_MAX_INT) PetscFunctionReturn(PETSC_SUCCESS);

  owner = (global_first >= offset && global_first < offset + nlocal) ? rank
                                                                      : INT_MAX;
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &owner, 1, MPI_INT, MPI_MIN,
                             PetscObjectComm((PetscObject)v)));
  if (global_first >= offset && global_first < offset + nlocal) {
    PetscCall(VecGetArrayRead(v, &array));
    pivot = array[global_first - offset];
    PetscCall(VecRestoreArrayRead(v, &array));
  }
  PetscCallMPI(
      MPI_Bcast(&pivot, 1, MPIU_SCALAR, owner, PetscObjectComm((PetscObject)v)));
  if (PetscRealPart(pivot) < 0.0) PetscCall(VecScale(v, -1.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv) {
  Mat A, B;
  EPS eps;
  ST st;
  Vec column;
  PetscBool flg;
  PetscInt nconv, seed = 0;

  PetscCall(SlepcInitialize(&argc, &argv, NULL, NULL));

  /* Create first matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 100, 100));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /* Create second matrix with same layout */
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &B));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-random_seed", &seed, &flg));
  if (!flg)
    seed = 0;

  /* Fill matrices from a seed and global indices so MPI partitioning does not
   * change the generated values. */
  PetscCall(FillMatrixDeterministically(A, seed, 0));
  PetscCall(FillMatrixDeterministically(B, seed, 1));

  PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
  PetscCall(EPSSetOperators(eps, A, B));
  PetscCall(EPSSetProblemType(eps, EPS_GHEP));
  PetscCall(EPSSetTarget(eps, 0.0));
  PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE));
  PetscCall(EPSGetST(eps, &st));
  PetscCall(STSetType(st, STSINVERT));
  PetscCall(EPSSetFromOptions(eps));
  PetscCall(EPSSolve(eps));
  PetscCall(EPSGetConverged(eps, &nconv));
  PetscCall(MatCreateVecs(A, &column, NULL));
  for (PetscInt i = 0; i < PetscMin((PetscInt)10, nconv); ++i) {
    char filename[256];

    PetscCall(PetscSNPrintf(filename, sizeof(filename), "col%" PetscInt_FMT, i));
    PetscCall(EPSGetEigenvector(eps, i, column, NULL));
    PetscCall(CanonicalizeEigenvectorSign(column));
    init_debug_viewer(PETSC_COMM_WORLD, filename);
    PetscCall(VecView(column, debug_viewer));
    destroy_debug_viewer();
  }

  /* View matrices */
  init_debug_viewer(PETSC_COMM_WORLD, "A");
  PetscCall(MatView(A, debug_viewer));
  destroy_debug_viewer();
  init_debug_viewer(PETSC_COMM_WORLD, "B");
  PetscCall(MatView(B, debug_viewer));
  destroy_debug_viewer();
  /* Cleanup */
  PetscCall(VecDestroy(&column));
  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));

  PetscCall(SlepcFinalize());
  return 0;
}
