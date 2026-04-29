#include "petscsys.h"
#include <petscis.h>
#include <petscksp.h>
#include <petscviewer.h>

/* Split a local size m into two child sizes, aligned on block boundaries.
   child = 0 gets the first chunk, child = 1 gets the remainder. */
static PetscErrorCode SplitLocalSizeByBlock(PetscInt m, PetscInt bs,
                                            PetscInt child, PetscInt *mchild,
                                            PetscInt *offset) {
  PetscInt total_blocks, child_0_blocks, child_1_blocks;

  PetscFunctionBegin;
  total_blocks = m / bs; 
  child_0_blocks = total_blocks / 2;        /* first child gets floor(nblocks/2) */
  child_1_blocks = total_blocks - child_0_blocks; /* second child gets the rest */
  *offset = child == 0? 0: child_0_blocks*bs; 
  *mchild = child == 0? child_0_blocks * bs: child_1_blocks*bs; 
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Build a PETSC_COMM_SELF IS containing a contiguous slice of an existing
   local array of global indices, preserving block size metadata. */
static PetscErrorCode
CreateChildGlobalISFromParentLocalIndices(const PetscInt parentIdx[],
                                          PetscInt mParent, PetscInt bs,
                                          PetscInt child, IS *isChild) {
  PetscInt offset, m_child;

  PetscFunctionBegin;
  PetscCall(SplitLocalSizeByBlock(mParent, bs, child, &m_child, &offset));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, m_child, parentIdx + offset,
                            PETSC_COPY_VALUES, isChild));
  PetscCall(ISSetBlockSize(*isChild, bs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Build the child auxiliary matrix by extracting a principal submatrix from
   the sequential parent auxiliary matrix.  The parent matrix is assumed to
   live on PETSC_COMM_SELF. */
static PetscErrorCode CreateChildAuxFromParentAux(Mat auxParent, PetscInt bs,
                                                  PetscInt child,
                                                  Mat *auxChild) {
  PetscInt m, n, offset, mchild;
  IS isloc;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(auxParent, &m, &n));
  PetscCall(SplitLocalSizeByBlock(m, bs, child, &mchild, &offset));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, mchild, offset, 1, &isloc));
  PetscCall(ISSetBlockSize(isloc, bs));
  PetscCall(MatCreateSubMatrix(auxParent, isloc, isloc, MAT_INITIAL_MATRIX,
                               auxChild));
  PetscCall(ISDestroy(&isloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args) {
  Mat A = NULL, Xseed = NULL, Bseed = NULL, auxParent = NULL, aux = NULL;
  Vec b = NULL;
  KSP ksp = NULL;
  PC pc = NULL;
  IS sizes = NULL, isSeed = NULL, is = NULL;
  PetscViewer viewer = NULL;
  PetscSubcomm psub = NULL;
  MPI_Comm subcomm = MPI_COMM_NULL;
  PetscLayout map = NULL;
  const PetscInt *idxSizes = NULL, *idxSeed = NULL;
  PetscMPIInt rank, size, subrank, subsize;
  PetscInt bsA = 1, bsAux = 1, nSeedLocal, mA, nA, MA, NA, mXseed;
  PetscBool flg;
  char dir[PETSC_MAX_PATH_LEN], name[PETSC_MAX_PATH_LEN], type[256];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, NULL));
  PetscCall(PetscLogDefaultBegin());
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCheck(size == 8, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE,
             "This adaptation example expects exactly 8 MPI ranks, got %d",
             (int)size);

  /* Parse options */
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "", "");
  PetscCall(PetscStrncpy(type, MATAIJ, sizeof(type)));
  PetscCall(PetscOptionsFList("-mat_type", "Matrix type", "MatSetType", MatList,
                              type, type, sizeof(type), &flg));
  PetscOptionsEnd();

  PetscCall(PetscStrncpy(dir, ".", sizeof(dir)));
  PetscCall(
      PetscOptionsGetString(NULL, NULL, "-load_dir", dir, sizeof(dir), NULL));

  /* --------------------------------------------------------------------- */
  /* 1) Load the global matrix A on 8 ranks. Do NOT prescribe local sizes
        from sizes_4.dat; let PETSc use a valid 8-rank distribution. */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(PetscSNPrintf(name, sizeof(name), "%s/A.dat", dir));
  PetscCall(
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
  PetscCall(MatLoad(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatConvert(A, type, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatGetBlockSize(A, &bsA));
  PetscCall(MatGetLocalSize(A, &mA, &nA));
  PetscCall(MatGetSize(A, &MA, &NA));

  PetscCall(PetscSynchronizedPrintf(
      PETSC_COMM_WORLD,
      "[world %d] A local=(%" PetscInt_FMT ",%" PetscInt_FMT
      ") global=(%" PetscInt_FMT ",%" PetscInt_FMT ") bs=%" PetscInt_FMT "\n",
      (int)rank, mA, nA, MA, NA, bsA));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

  /* --------------------------------------------------------------------- */
  /* 2) Split 8 world ranks into two 4-rank subcommunicators.
        subcomm 0: world ranks 0,1,2,3
        subcomm 1: world ranks 4,5,6,7
     Hence world ranks r and r+4 correspond to the same old parent subdomain. */
  PetscCall(PetscSubcommCreate(PETSC_COMM_WORLD, &psub));
  PetscCall(PetscSubcommSetNumber(psub, 2));
  PetscCall(PetscSubcommSetType(psub, PETSC_SUBCOMM_CONTIGUOUS));
  PetscCall(PetscSubcommGetChild(psub, &subcomm));

  PetscCallMPI(MPI_Comm_rank(subcomm, &subrank));
  PetscCallMPI(MPI_Comm_size(subcomm, &subsize));
  PetscCheck(subsize == 4, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Expected child subcommunicator size 4, got %d", (int)subsize);

  /* child selector inside each old parent: ranks 0..3 in world are child 0,
     ranks 4..7 in world are child 1. */
  {
    PetscInt child = (rank < 4) ? 0 : 1;

    /* ------------------------------------------------------------------- */
    /* 3) Load sizes_4.dat on each 4-rank subcommunicator.
          This file is only used as seed metadata for Xseed layout. */
    PetscCall(PetscSNPrintf(name, sizeof(name), "%s/sizes_4.dat", dir));
    PetscCall(PetscViewerBinaryOpen(subcomm, name, FILE_MODE_READ, &viewer));
    PetscCall(ISCreate(subcomm, &sizes));
    PetscCall(ISLoad(sizes, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(ISGetIndices(sizes, &idxSizes));

    PetscCheck(ISGetLocalSize(sizes, &nSeedLocal) == PETSC_SUCCESS,
               PETSC_COMM_WORLD, PETSC_ERR_PLIB,
               "ISGetLocalSize failed unexpectedly");
    PetscCall(ISGetLocalSize(sizes, &nSeedLocal));
    PetscCheck(nSeedLocal >= 5, subcomm, PETSC_ERR_ARG_SIZ,
               "sizes_4.dat must provide at least 5 local integers per rank, "
               "got %" PetscInt_FMT,
               nSeedLocal);

    /* Original example semantics:
         idxSizes[0] local rows of A on old 4-rank run
         idxSizes[1] local cols of A on old 4-rank run
         idxSizes[2] global rows of A
         idxSizes[3] global cols of A
         idxSizes[4] local size for Xseed/Neumann on old 4-rank run
    */
    mXseed = idxSizes[4];

    /* ------------------------------------------------------------------- */
    /* 4) Load old distributed is_4.dat on the 4-rank subcommunicator, then
          on each world rank create the child local IS by splitting the
          inherited parent local indices on block boundaries. */
    PetscCall(PetscSNPrintf(name, sizeof(name), "%s/is_4.dat", dir));
    PetscCall(PetscViewerBinaryOpen(subcomm, name, FILE_MODE_READ, &viewer));
    PetscCall(ISCreate(subcomm, &isSeed));

    /* Mirror the original code: bind the IS layout to Xseed's row layout. */
    PetscCall(MatCreate(subcomm, &Xseed));
    PetscCall(
        MatSetSizes(Xseed, mXseed, mXseed, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetUp(Xseed));
    PetscCall(MatGetLayouts(Xseed, &map, NULL));
    PetscCall(ISSetLayout(isSeed, map));

    PetscCall(ISLoad(isSeed, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(ISGetLocalSize(isSeed, &nSeedLocal));
    PetscCall(ISGetIndices(isSeed, &idxSeed));

    PetscCheck(nSeedLocal % bsA == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "Local parent IS size %" PetscInt_FMT
               " is not divisible by A block size %" PetscInt_FMT,
               nSeedLocal, bsA);

    PetscCall(CreateChildGlobalISFromParentLocalIndices(idxSeed, nSeedLocal,
                                                        bsA, child, &is));

    /* ------------------------------------------------------------------- */
    /* 5) Load Neumann_4.dat on the 4-rank subcommunicator, get the parent
          local diagonal block, duplicate it, then split the sequential parent
          auxiliary matrix into the child auxiliary matrix. */
    PetscCall(PetscSNPrintf(name, sizeof(name), "%s/Neumann_4.dat", dir));
    PetscCall(PetscViewerBinaryOpen(subcomm, name, FILE_MODE_READ, &viewer));
    PetscCall(MatLoad(Xseed, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(MatGetDiagonalBlock(Xseed, &Bseed)); /* borrowed from Xseed */
    PetscCall(MatDuplicate(Bseed, MAT_COPY_VALUES,
                           &auxParent)); /* owned, sequential */
    PetscCall(MatConvert(auxParent, type, MAT_INPLACE_MATRIX, &auxParent));
    PetscCall(MatGetBlockSize(auxParent, &bsAux));

    /* We want the child auxiliary to be split consistently with the global
       block size of A. If auxParent has a different block size metadata, keep
       A's block size as the splitting rule and then set metadata afterward. */
    PetscCall(CreateChildAuxFromParentAux(auxParent, bsA, child, &aux));
    PetscCall(MatSetBlockSize(aux, bsA));
    PetscCall(MatConvert(aux, type, MAT_INPLACE_MATRIX, &aux));

    PetscCall(PetscSynchronizedPrintf(
        PETSC_COMM_WORLD,
        "[world %d | sub %d | child %d] parent is local=%" PetscInt_FMT
        ", auxParent bs=%" PetscInt_FMT "\n",
        (int)rank, (int)subrank, (int)child, nSeedLocal, bsAux));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

    PetscCall(ISRestoreIndices(isSeed, &idxSeed));
    PetscCall(ISRestoreIndices(sizes, &idxSizes));
  }

  /* Old seed objects no longer needed */
  PetscCall(ISDestroy(&sizes));
  PetscCall(ISDestroy(&isSeed));
  PetscCall(MatDestroy(&auxParent));
  PetscCall(MatDestroy(&Xseed));
  PetscCall(PetscSubcommDestroy(&psub));

  /* --------------------------------------------------------------------- */
  /* 6) Build RHS and solve with HPDDM using the child local (is, aux). */
  PetscCall(MatCreateVecs(A, NULL, &b));
  PetscCall(VecSet(b, 1.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCHPDDM));
  PetscCall(PCHPDDMSetAuxiliaryMat(pc, is, aux, NULL, NULL));

  /* User references can go away after setter call */
  PetscCall(ISDestroy(&is));
  PetscCall(MatDestroy(&aux));

  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, b));

  /* Optional summary */
  {
    KSPConvergedReason reason;
    PetscInt its;
    PetscReal rnorm;

    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "KSPSolve done: its=%" PetscInt_FMT
                          ", reason=%d, final rnorm=%g\n",
                          its, (int)reason, (double)rnorm));
  }
  char cmp_sol[PETSC_MAX_PATH_LEN];
  PetscBool cmp_flg = PETSC_FALSE;

  PetscCall(PetscOptionsGetString(NULL, NULL, "-compare_solution", cmp_sol,
                                  sizeof(cmp_sol), &cmp_flg));
  if (1) {
    Vec xref, diff;
    PetscViewer v;
    PetscReal ndiff, nxref, rel;

    PetscCall(VecDuplicate(b, &xref));
    PetscCall(
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, "x_ref_4.dat", FILE_MODE_READ, &v));
    PetscCall(VecLoad(xref, v));
    PetscCall(PetscViewerDestroy(&v));

    PetscCall(VecDuplicate(b, &diff));
    PetscCall(VecCopy(b, diff));
    PetscCall(VecAXPY(diff, -1.0, xref));

    PetscCall(VecNorm(diff, NORM_2, &ndiff));
    PetscCall(VecNorm(xref, NORM_2, &nxref));
    rel = (nxref > 0.0) ? ndiff / nxref : ndiff;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "Comparison against %s: abs=%g rel=%g\n", cmp_sol,
                          (double)ndiff, (double)rel));

    PetscCall(VecDestroy(&diff));
    PetscCall(VecDestroy(&xref));
  }
  PetscCall(VecDestroy(&b));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
