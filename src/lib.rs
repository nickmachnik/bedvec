//! A sparse representation of biallelic genotype data (e.g. as loaded from PLINK .bed files).
#![cfg_attr(feature = "unstable", feature(test))]

extern crate blas_src;
extern crate openblas_src;

pub mod bedvec;
