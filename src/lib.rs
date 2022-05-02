//! A sparse representation of biallelic genotype data (e.g. as loaded from PLINK .bed files).

mod bed_lookup_tables;
pub mod bedvec;
pub mod io;
mod simd;
