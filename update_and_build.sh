#!/bin/bash

set -Eeuo pipefail;
trap 'echo "Error on line $LINENO"; exit 1' ERR;
ROOT_DIR="$(pwd)"

if [ -e ".envfile" ]; then
    source .envfile;
fi
PETSC_DIR=${PETSC_DIR:-""}
if [ -z $PETSC_DIR ]; then
    echo "ERROR: PETSC_DIR unset. Stopping...";
    exit 1;
fi
PETSC_ARCH=${PETSC_ARCH:-""}
if [ -z $PETSC_ARCH ]; then
    echo "ERROR: PETSC_ARCH unset. Stopping...";
    exit 1;
fi
HPDDM_DEV_DIR=${HPDDM_DEV_DIR:-""}
if [ -z $HPDDM_DEV_DIR ]; then
    echo "ERROR: HPDDM_DEV_DIR unset. Stopping...";
    exit 1;
fi

OPTIONS_SKIP_UPDATE=0
OPTIONS_SKIP_RUNNERS=0
OPTIONS_SKIP_PETSC=0
for flag in $@; do
    if [ "-" == "${flag:0:1}" ]; then
        case "$flag" in
            "-u")
                OPTIONS_SKIP_UPDATE=1
                ;;
            "-r")
                OPTIONS_SKIP_RUNNERS=1
                ;;
            "-p")
                OPTIONS_SKIP_PETSC=1
                ;;
            *)
                echo "Warning: unrecognized flag ($flag), ignoring.."
                ;;
        esac
    else
        echo "Warning: unrecognized parameter passed ($flag), ignoring..";
    fi
done

if [ "$OPTIONS_SKIP_UPDATE" -eq 0 ]; then
    HPDDM_HEADER_SOURCE="$HPDDM_DEV_DIR/include"
    PETSC_ARCH_HEADER_SOURCE="$PETSC_DIR/$PETSC_ARCH/include"
    SOURCE_DIFF="$(diff -q "$HPDDM_HEADER_SOURCE" "$PETSC_ARCH_HEADER_SOURCE"| grep -v 'Only in' || true)"
    if ! [ -z "$SOURCE_DIFF" ]; then
        echo "Change in headers detected..."
        echo "I will update headers with the following:"
        echo -e "\tSRC:$HPDDM_HEADER_SOURCE"
        echo -e "\tDST:$PETSC_ARCH_HEADER_SOURCE"
        read -p "Do you want to continue? [y/n] " answer
        case "$answer" in
            y|Y ) cp $HPDDM_HEADER_SOURCE/* $PETSC_ARCH_HEADER_SOURCE ;;
            * ) echo "Aborting..."; exit 1 ;;
        esac
    fi

    KSPHPDDM_HPDDM_SOURCE="$HPDDM_DEV_DIR/interface/petsc/ksp"
    KSPHPDDM_PETSC_SOURCE="$PETSC_DIR/src/ksp/ksp/impls/hpddm"
    SOURCE_DIFF="$(diff -rq $KSPHPDDM_HPDDM_SOURCE $KSPHPDDM_PETSC_SOURCE | grep -v "makefile" || true)"
    if ! [ -z "$SOURCE_DIFF" ]; then
        echo "Change in KSPHPDDM detected..."
        echo "I will update headers with the following:"
        echo -e "\tSRC:$KSPHPDDM_HPDDM_SOURCE"
        echo -e "\tDST:$KSPHPDDM_PETSC_SOURCE"
        read -p "Do you want to continue? [y/n] " answer
        case "$answer" in
            y|Y ) cp -r $KSPHPDDM_HPDDM_SOURCE/* $KSPHPDDM_PETSC_SOURCE ;;
            * ) echo "Aborting..."; exit 1 ;;
        esac
    fi

    PCHPDDM_HPDDM_SOURCE="$HPDDM_DEV_DIR/interface/petsc/pc"
    PCHPDDM_PETSC_SOURCE="$PETSC_DIR/src/ksp/pc/impls/hpddm"
    SOURCE_DIFF="$(diff -rq $PCHPDDM_HPDDM_SOURCE $PCHPDDM_PETSC_SOURCE|grep -Ev "makefile|ftn-custom" || true)"
    if ! [ -z "$SOURCE_DIFF" ]; then
        echo "Change in PCHPDDM detected..."
        echo "I will update headers with the following:"
        echo -e "\tSRC:$PCHPDDM_HPDDM_SOURCE"
        echo -e "\tDST:$PCHPDDM_PETSC_SOURCE"
        read -p "Do you want to continue? [y/n] " answer
        case "$answer" in
            y|Y ) cp -r $PCHPDDM_HPDDM_SOURCE/* $PCHPDDM_PETSC_SOURCE ;;
            * ) echo "Aborting..."; exit 1 ;;
        esac
    fi
fi

TEMP_FILE=$(mktemp);
if [ $OPTIONS_SKIP_PETSC -eq 0 ]; then
    echo "Building PETSc...";
    cd $PETSC_DIR;
    make libs hpddmbuild > "$TEMP_FILE" 2>&1 || { cat "$TEMP_FILE" && exit 1; };
    cd "$ROOT_DIR";
    echo "Success!";
fi

if [ $OPTIONS_SKIP_RUNNERS -eq 0 ]; then
    ROOT_DIR="$(pwd)"
    echo "Building runners..."
    if [ ! -d "./tests/" ]; then
        echo "Error: no test directory detected. Stopping..";
        exit 1;
    fi
    BIN_DIR="$ROOT_DIR/bin/$PETSC_ARCH"
    if ! [ -d "$BIN_DIR" ]; then
        mkdir -p "$BIN_DIR"
    fi
    cd "./tests/"
    TEST_SOURCES=(*.c);
    for SOURCE in "${TEST_SOURCES[@]}"; do
        STRIPPED="${SOURCE%.c}"
        echo -e "\t -Building $STRIPPED..."
        if [ -e "$STRIPPED" ]; then
            rm "$STRIPPED"
        fi
        make "$STRIPPED" > "$TEMP_FILE" 2>&1 || { cat "$TEMP_FILE" && exit 1; };
        mv "$STRIPPED" "$BIN_DIR/$STRIPPED"
    done
    cd "$ROOT_DIR";
    echo "Success!";
fi
rm $TEMP_FILE


