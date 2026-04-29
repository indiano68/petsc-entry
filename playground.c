const char help[] = "Test MatCreateDenseFromVecType()\n\n";

#include <assert.h>
#include <petscconf.h>
#include <petscdevice_cuda.h>
#include <petscmat.h>

int main(int argc, char **args) {
  Mat A, B;
  Vec X;
  VecType vtype;
  PetscInt n = 20, lda = PETSC_DECIDE;
  PetscBool use_memtype = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL,
                    "Creating Mat from Vec type example", NULL);
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-lda", &lda, NULL));
  PetscCall(
      PetscOptionsGetBool(NULL, NULL, "-use_memtype", &use_memtype, NULL));
  PetscOptionsEnd();
  if (lda > 0)
    lda += n;

  PetscCall(VecCreate(PETSC_COMM_WORLD, &X));
  PetscCall(VecSetSizes(X, n, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(X));
  PetscCall(VecSetUp(X));
  PetscCall(VecGetType(X, &vtype));

  PetscCall(MatCreateDenseFromVecType(PETSC_COMM_WORLD, vtype, n, n,
                                      PETSC_DECIDE, PETSC_DECIDE, lda, NULL,
                                      &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscRandom rctx;
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetSeed(rctx, 12345));
  PetscCall(PetscRandomSeed(rctx));
  PetscCall(MatSetRandom(A, rctx));
  PetscCall(PetscRandomDestroy(&rctx));

  PetscMemType X_memtype, A_memtype;
  PetscScalar *array;
  PetscCall(VecGetArrayAndMemType(X, &array, &X_memtype));
  PetscCall(VecRestoreArrayAndMemType(X, &array));
  PetscCall(MatDenseGetArrayAndMemType(A, &array, &A_memtype));
  if (use_memtype)
    PetscCall(MatCreateDenseFromMemType(PETSC_COMM_WORLD, A_memtype, n, n,
                                        PETSC_DECIDE, PETSC_DECIDE, lda, array,
                                        &B));
  PetscCall(MatDenseRestoreArrayAndMemType(A, &array));
  PetscAssert(A_memtype == X_memtype, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
              "Failed memtype guarantee in MatCreateDenseFromVecType");

  /* test */
  PetscCall(MatViewFromOptions(use_memtype ? B : A, NULL, "-ex19_mat_view"));
  if (use_memtype)
    PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  testset:
    args: -lda {{0 1}} -ex19_mat_view -use_memtype {{true false}}
    filter: grep -v -i type
    output_file: output/ex19.out

    test:
      suffix: cuda
      requires: cuda
      args: -vec_type {{cuda mpicuda}}

    test:
      suffix: hip
      requires: hip
      args: -vec_type hip

    test:
      suffix: standard
      args: -vec_type standard

    test:
      suffix: kokkos
      # we don't have MATDENSESYCL yet
      requires: kokkos_kernels !sycl
      args: -vec_type kokkos
TEST*/
