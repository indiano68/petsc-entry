#include "petscis.h"
#include <petsc.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Mat A, X, aux;
  IS sizes_is, is;
  const PetscInt *indexes;
  PetscLayout map;
  Mat *sub;

  PetscViewer viewer;
  char datadir[PETSC_MAX_PATH_LEN] ,
       filename[PETSC_MAX_PATH_LEN];

  PetscSubcomm subcomm_read, subcomm_work;
  PetscMPIInt global_rank, global_size;
  MPI_Comm mpisubcomm_work, mpisubcomm_read;
  PetscInt n_subdomains = 4, n_subcomms_read;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-load_dir", datadir, sizeof(datadir), NULL));
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
  /****************************************************************************/

  /********************************** Cre.is **********************************/
  PetscCall(PetscSNPrintf(filename, sizeof(filename), "%s/is_4.dat", datadir));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ,
                                  &viewer));
  PetscCall(ISCreate(PETSC_COMM_WORLD, &sizes_is));
  PetscCall(MatGetLayouts(X, &map, NULL));
  PetscCall(ISSetLayout(sizes_is, map));
  PetscCall(ISLoad(sizes_is, viewer));
  PetscCall(ISSetBlockSize(sizes_is, bs));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(ISSetBlockSize(sizes_is, bs));
  PetscCall(ISOnComm(sizes_is, mpisubcomm_work, PETSC_COPY_VALUES, &is));
  PetscCall(ISDestroy(&sizes_is));
  PetscCall(MatDestroy(&X));

  /****************************************************************************/


  PetscCall(MatCreateSubMatricesMPI(A, 1, &is, &is, MAT_INITIAL_MATRIX, &sub));
  PetscInt subcom_size = global_size /n_subdomains;
  PetscInt color = global_rank / subcom_size;
  snprintf(filename, sizeof(filename),
           "A_subcom%d_%d.dat", color, global_size);
  PetscCall(PetscViewerASCIIOpen(mpisubcomm_work, filename, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(MatView(sub[0], viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /********************************** Clean **********************************/
  PetscCall(PetscSubcommDestroy(&subcomm_read));
  PetscCall(PetscSubcommDestroy(&subcomm_work));
  PetscCall(ISDestroy(&is));
  PetscCall(MatDestroy(&A));
  /****************************************************************************/
  PetscCall(PetscFinalize());
}
