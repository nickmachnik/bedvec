//! A sparse representation of biallelic genotype data (e.g. as loaded from PLINK .bed files).
extern crate blas_src;
extern crate openblas_src;

mod bed_lookup_tables;
pub mod bedvec;
