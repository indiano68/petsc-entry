#include "mpi.h"
#include "mpi_proto.h"
#include "petscconf.h"
#include "petscis.h"
#include "petscistypes.h"
#include "petscksp.h"
#include "petsclog.h"
#include "petscmat.h"
#include "petscpc.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscviewer.h"
#include <HPDDM_debug.hpp>
#include <petsc.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Mat A, X, aux;
  KSP ksp;
  PC pc;
  IS sizes_is, is;
  const PetscInt *indexes;
  PetscLayout map;

  PetscViewer viewer;
  char datadir[PETSC_MAX_PATH_LEN] = "/Users/erikfabrizzi/Workspace/LIP6-SWE/"
                                     "repos/datafiles/matrices/hpddm/GENEO",
       filename[PETSC_MAX_PATH_LEN];

  PetscSubcomm subcomm_read, subcomm_work;
  PetscMPIInt global_rank, global_size;
  MPI_Comm mpisubcomm_work, mpisubcomm_read;
  PetscInt n_subdomains = 4, n_subcomms_read;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &global_rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &global_size));
  PetscCheck(global_size % n_subdomains == 0, PETSC_COMM_WORLD,
             PETSC_ERR_WRONG_MPI_SIZE, "This example requires n*%d processes",
             n_subdomains);
  PetscCall(PetscSubcommCreate(PETSC_COMM_WORLD, &subcomm_read));
  n_subcomms_read = (global_size + n_subdomains - 1) / n_subdomains;
  PetscCall(PetscSubcommSetNumber(subcomm_read, n_subcomms_read));
  PetscCall(PetscSubcommSetType(subcomm_read, PETSC_SUBCOMM_INTERLACED));
  PetscCall(PetscSubcommGetChild(subcomm_read, &mpisubcomm_read));

  PetscCall(PetscSubcommCreate(PETSC_COMM_WORLD, &subcomm_work));
  PetscCall(PetscSubcommSetNumber(subcomm_work, n_subdomains));
  PetscCall(PetscSubcommSetType(subcomm_work, PETSC_SUBCOMM_CONTIGUOUS));
  PetscCall(PetscSubcommGetChild(subcomm_work, &mpisubcomm_work));

  /********************************* Load is **********************************/
  if (global_rank % n_subcomms_read == 0) {
    PetscCall(
        PetscSNPrintf(filename, sizeof(filename), "%s/sizes_4.dat", datadir));
    PetscCall(PetscViewerBinaryOpen(mpisubcomm_read, filename, FILE_MODE_READ,
                                    &viewer));
    PetscCall(ISCreate(mpisubcomm_read, &sizes_is));
    PetscCall(ISLoad(sizes_is, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(ISGetIndices(sizes_is, &indexes));
  } else
    indexes = malloc(sizeof(PetscInt) * 5);
  PetscCallMPI(MPI_Bcast((void *)indexes, 5, MPIU_INT, 0, mpisubcomm_work));
  /****************************************************************************/

  /********************************** Load A **********************************/
  PetscInt n = PETSC_DECIDE, m = PETSC_DECIDE, bs = 2;
  PetscCall(PetscSplitOwnershipBlock(mpisubcomm_work, bs, &n, indexes + 1));
  PetscCall(PetscSplitOwnershipBlock(mpisubcomm_work, bs, &m, indexes));
  PetscCall(PetscSNPrintf(filename, sizeof(filename), "%s/A.dat", datadir));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ,
                                  &viewer));
  PetscCall(MatSetSizes(A, m, n, indexes[2], indexes[3]));
  PetscCall(MatLoad(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  /****************************************************************************/

  /********************************** Load X **********************************/
  n = PETSC_DECIDE;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &X));
  PetscCall(PetscSplitOwnershipBlock(mpisubcomm_work, bs, &n, indexes + 4));
  PetscCall(MatSetSizes(X, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetUp(X));
  PetscCall(
      PetscSNPrintf(filename, sizeof(filename), "%s/Neumann_4.dat", datadir));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ,
                                  &viewer));
  PetscCall(MatLoad(X, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatGetMultiProcBlock(X, mpisubcomm_work, MAT_INITIAL_MATRIX, &aux));
  PetscCall(MatConvert(aux, MATAIJ, MAT_INPLACE_MATRIX, &aux));
  
  PetscCall(PetscViewerDestroy(&viewer));
  if (global_rank % n_subcomms_read == 0) {
    PetscCall(ISRestoreIndices(sizes_is, &indexes));
    PetscCall(ISDestroy(&sizes_is));
  } else
    free((void *)indexes);

  init_debug_viewer(PetscObjectComm((PetscObject)aux), "aux");
  MatView(aux, debug_viewer);
  destroy_debug_viewer();

  /****************************************************************************/

  /********************************** Cre.is **********************************/
  char pattern[256];
  int work_size;


  PetscCall(PetscSNPrintf(filename, sizeof(filename), "%s/is_4.dat", datadir));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ,
                                  &viewer));
  PetscCall(ISCreate(PETSC_COMM_WORLD, &sizes_is));
  PetscCall(MatGetLayouts(X, &map, NULL));
  PetscCall(ISSetLayout(sizes_is, map));
  PetscCall(ISLoad(sizes_is, viewer));
  PetscCall(ISSetBlockSize(sizes_is, bs));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(ISDuplicate(sizes_is, &is));
  PetscCall(ISSetBlockSize(sizes_is, bs));
  PetscCall(ISOnComm(sizes_is, mpisubcomm_work, PETSC_COPY_VALUES, &is));
  PetscCall(ISDestroy(&sizes_is));
  PetscCall(MatDestroy(&X));

  init_debug_viewer(PetscObjectComm((PetscObject)is), "is");
  ISView(is, debug_viewer);
  destroy_debug_viewer();

  /****************************************************************************/

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCHPDDM));
  PetscCall(PCHPDDMSetAuxiliaryMat(pc, is, aux, NULL, NULL));
  PetscCall(MatDestroy(&aux));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(ISDestroy(&is));

  /********************************** Clean **********************************/
  PetscCall(PetscSubcommDestroy(&subcomm_read));
  PetscCall(PetscSubcommDestroy(&subcomm_work));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  /****************************************************************************/
  PetscCall(PetscFinalize());
}

/*TEST

   test:
      suffix: clean
      output_file: output/empty.out
      requires: hpddm slepc datafilespath double !complex
!defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
defined(PETSC_USE_SHARED_LIBRARIES) nsize: 8 args: -ksp_converged_reason
-ksp_atol 1e-8

   test:
      requires: hpddm slepc datafilespath double !complex
!defined(PETSC_USE_64BIT_INDICES) defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
defined(PETSC_USE_SHARED_LIBRARIES) output_file: output/empty.out nsize: 8 args:
-ksp_converged_reason -pc_hpddm_levels_1_eps_nev 2 -ksp_atol 1e-8

TEST*/
