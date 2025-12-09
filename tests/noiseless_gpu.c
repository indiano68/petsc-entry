#include "petscsys.h"
#include <petscksp.h>
#include <petsc/private/petscimpl.h>

static char help[] = "Solves a linear system using PCHPDDM.\n\n";

int main(int argc, char **args)
{
  Vec             b;            /* computed solution and RHS */
  Mat             A, aux, X, B; /* linear system matrix */
  KSP             ksp;          /* linear solver context */
  PC              pc;
  IS              is, sizes;
  const PetscInt *idx;
  PetscMPIInt     rank, size;
  PetscInt        m, N = 1;
  PetscLayout     map;
  PetscViewer     viewer;
  char            dir[PETSC_MAX_PATH_LEN], name[PETSC_MAX_PATH_LEN], type[256];
  PetscBool       flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscLogDefaultBegin());
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 4, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This example requires 4 processes");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-rhs", &N, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(PetscStrncpy(dir, ".", sizeof(dir)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-load_dir", dir, sizeof(dir), NULL));

  /* loading matrices */
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s/sizes_%d.dat", dir, size));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
  PetscCall(ISCreate(PETSC_COMM_WORLD, &sizes));
  PetscCall(ISLoad(sizes, viewer));
  PetscCall(ISGetIndices(sizes, &idx));
  PetscCall(MatSetSizes(A, idx[0], idx[1], idx[2], idx[3]));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &X));
  PetscCall(MatSetSizes(X, idx[4], idx[4], PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetUp(X));
  PetscCall(ISRestoreIndices(sizes, &idx));
  PetscCall(ISDestroy(&sizes));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s/A.dat", dir));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
  PetscCall(MatLoad(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s/is_%d.dat", dir, size));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
  PetscCall(ISCreate(PETSC_COMM_WORLD, &sizes));
  PetscCall(MatGetLayouts(X, &map, NULL));
  PetscCall(ISSetLayout(sizes, map));
  PetscCall(ISLoad(sizes, viewer));
  PetscCall(ISGetLocalSize(sizes, &m));
  PetscCall(ISGetIndices(sizes, &idx));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, m, idx, PETSC_COPY_VALUES, &is));
  PetscCall(ISRestoreIndices(sizes, &idx));
  PetscCall(ISDestroy(&sizes));
  PetscCall(MatGetBlockSize(A, &m));
  PetscCall(ISSetBlockSize(is, m));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s/Neumann_%d.dat", dir, size));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
  PetscCall(MatLoad(X, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatGetDiagonalBlock(X, &B));
  PetscCall(MatDuplicate(B, MAT_COPY_VALUES, &aux));
  PetscCall(MatDestroy(&X));
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(aux, MAT_SYMMETRIC, PETSC_TRUE));

  /* ready for testing */
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "", "");
  PetscCall(PetscStrncpy(type, MATAIJCUSPARSE, sizeof(type)));
  PetscCall(PetscOptionsFList("-mat_type", "Matrix type", "MatSetType", MatList, type, type, 256, &flg));
  PetscOptionsEnd();
  PetscCall(MatConvert(A, type, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatConvert(aux, type, MAT_INPLACE_MATRIX, &aux));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCHPDDM));
  PetscCall(PCHPDDMSetAuxiliaryMat(pc, is, aux, NULL, NULL));
  PetscCall(PCHPDDMHasNeumannMat(pc, PETSC_TRUE)); /* PETSC_TRUE is fine as well, just testing */
  PetscCall(MatDestroy(&aux));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(ISDestroy(&is));
  PetscCall(MatCreateVecs(A, NULL, &b));
  PetscCall(VecSet(b, 1.0));
  PetscCall(KSPSolve(ksp, b, b));
  PetscCall(VecGetLocalSize(b, &m));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCHPDDM, &flg));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
   test:
      requires: hpddm slepc datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      suffix: fgmres_geneo_20_p_2
      nsize: 4
      args: -ksp_converged_reason -pc_type hpddm -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_pc_type redundant -ksp_type fgmres -pc_hpddm_coarse_mat_type baij -pc_hpddm_log_separate false  -load_dir ${DATAFILESPATH}/matrices/hpddm/GENEO
TEST*/
