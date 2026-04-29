#include "petscsys.h"
#include <petsc/private/petscimpl.h>
#include <petscksp.h>

static char help[] = "Solves a linear system using PCHPDDM.\n\n";

int main(int argc, char **args) {
  Vec b;            /* computed solution and RHS */
  Mat A, aux, X, B; /* linear system matrix */
  KSP ksp;          /* linear solver context */
  PC pc;
  IS is, sizes;
  const PetscInt *idx;
  PetscMPIInt size;
  PetscInt m;
  PetscLayout map;
  PetscViewer viewer;
  char dir[PETSC_MAX_PATH_LEN], name[PETSC_MAX_PATH_LEN], type[256];
  PetscBool flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscLogDefaultBegin());
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCheck(
      size == 4, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE,
      "This example requires 4 processes"); // early out here, needs to go.

  // Parses mat type
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "", "");
  PetscCall(PetscStrncpy(type, MATAIJ, sizeof(type)));
  PetscCall(PetscOptionsFList("-mat_type", "Matrix type", "MatSetType", MatList,
                              type, type, 256, &flg));
  PetscOptionsEnd();

  // Parses load dir
  PetscCall(PetscStrncpy(dir, ".", sizeof(dir)));
  PetscCall(
      PetscOptionsGetString(NULL, NULL, "-load_dir", dir, sizeof(dir), NULL));

  /* Get A/X indexes */
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s/sizes_%d.dat", dir, size));
  PetscCall(
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
  PetscCall(ISCreate(PETSC_COMM_WORLD, &sizes));
  PetscCall(ISLoad(sizes, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(ISGetIndices(sizes, &idx));

  /* Crate A */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, idx[0], idx[1], idx[2], idx[3]));
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s/A.dat", dir));
  PetscCall(
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
  PetscCall(MatLoad(A, viewer));
  PetscCall(MatConvert(A, type, MAT_INPLACE_MATRIX, &A));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Create X */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &X));
  PetscCall(MatSetSizes(X, idx[4], idx[4], PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetUp(X));

  /* Clean A/X indexes */
  PetscCall(ISRestoreIndices(sizes, &idx));
  PetscCall(ISDestroy(&sizes));

  /* Create IS */
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s/is_%d.dat", dir, size));
  PetscCall(
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
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

  /* Create AUX */
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s/Neumann_%d.dat", dir, size));
  PetscCall(
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
  PetscCall(MatLoad(X, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatGetDiagonalBlock(
      X, &B)); // B is a buffer for the diagonal block-> copied into aux, can
               // not substitute aux here or it gets destroyed with X , also no
               // need for MatDestroy(&B), goes with X
  PetscCall(MatDuplicate(B, MAT_COPY_VALUES, &aux));
  PetscCall(MatConvert(aux, type, MAT_INPLACE_MATRIX, &aux));
  PetscCall(MatDestroy(&X));

  /* Create RHS */
  PetscCall(MatCreateVecs(A, NULL, &b));
  PetscCall(VecSet(b, 1.0));

  /* Setup KSP/PC */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCHPDDM));
  PetscCall(PCHPDDMSetAuxiliaryMat(pc, is, aux, NULL, NULL));
  PetscCall(MatDestroy(&aux));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(ISDestroy(&is));

  /* Solve */
  PetscCall(KSPSolve(ksp, b, b));
  PetscViewer viewer_sol;

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "x_ref_4.dat",
                                  FILE_MODE_WRITE, &viewer_sol));
  PetscCall(VecView(b, viewer_sol));
  PetscCall(PetscViewerDestroy(&viewer_sol));
  PetscCall(VecDestroy(&b));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
