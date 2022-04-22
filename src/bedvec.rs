use crate::bed_lookup_tables::*;
use rand::distributions::Distribution;
use rand::Rng;
use rayon::prelude::*;
use statrs::distribution::Binomial;

/// Row-major bed-data in memory.
pub struct BinBedVecRM {
    data: Vec<u8>,
    num_individuals: usize,
    num_markers: usize,
    row_padding_bits: usize,
    bytes_per_row: usize,
}

impl BinBedVecRM {
    pub fn new(data: Vec<u8>, num_individuals: usize, num_markers: usize) -> Self {
        let row_padding_bits = (num_markers % 4) * 2;
        let bytes_per_row = if row_padding_bits == 0 {
            num_markers / 4
        } else {
            num_markers / 4 + 1
        };
        Self {
            data,
            num_individuals,
            num_markers,
            row_padding_bits,
            bytes_per_row,
        }
    }

    /// Create a completely random new BinBedVecRM of given dimensions.
    pub fn new_rnd(num_individuals: usize, num_markers: usize) -> Self {
        let mut rng = rand::thread_rng();
        let row_padding_bits = (num_markers % 4) * 2;
        let bytes_per_row = if row_padding_bits == 0 {
            num_markers / 4
        } else {
            num_markers / 4 + 1
        };
        let data: Vec<u8> = (0..(num_individuals * bytes_per_row))
            .map(|_| rng.gen())
            .collect();
        Self {
            data,
            num_individuals,
            num_markers,
            row_padding_bits,
            bytes_per_row,
        }
    }

    pub fn mul_with_vec(&self, v: &[f32]) -> Vec<f32> {
        (0..self.num_individuals)
            .into_par_iter()
            .map(|row_ix| self.row_dot_product(row_ix, v))
            .collect()
    }

    // TODO: this doesn't do standardization yet.
    // It also doesn't take into account the number of padding bits.
    // Ideally these will be always NAN, s.t. they are zero in the end anyway.
    #[inline(always)]
    fn row_dot_product(&self, row_ix: usize, v: &[f32]) -> f32 {
        let start_ix = row_ix * self.bytes_per_row;
        self.data[start_ix..start_ix + self.bytes_per_row]
            .par_iter()
            .enumerate()
            .map(|(byte_ix, byte)| {
                let unpacked_byte = self.unpack_byte_to_genotype_and_validity(byte);
                let v_ix = byte_ix * 4;
                unpacked_byte[0] * unpacked_byte[4] * v[v_ix]
                    + unpacked_byte[1] * unpacked_byte[5] * v[v_ix + 1]
                    + unpacked_byte[2] * unpacked_byte[6] * v[v_ix + 2]
                    + unpacked_byte[3] * unpacked_byte[7] * v[v_ix + 3]
            })
            .sum()
    }

    #[inline(always)]
    fn unpack_byte_to_genotype_and_validity(&self, byte: &u8) -> [f32; 8] {
        let start_ix = *byte as usize * 8;
        BED_LOOKUP_GENOTYPE_AND_VALIDITY[start_ix..start_ix + 8]
            .try_into()
            .expect("Failed to unpack bed byte")
    }
}

pub struct BedVecContig {
    ixs: Vec<u32>,
    twos_from: usize,
}

impl BedVecContig {
    pub fn new(genotypes: &[u8]) -> Self {
        let mut ones = Vec::new();
        let mut twos = Vec::new();
        for (ix, g) in genotypes.iter().enumerate() {
            match g {
                1 => ones.push(ix as u32),
                2 => twos.push(ix as u32),
                _ => continue,
            }
        }
        let twos_from = ones.len();
        twos.iter().for_each(|e| ones.push(*e));
        Self {
            ixs: ones,
            twos_from,
        }
    }

    pub fn scaled_add(&self, lhs: &mut [f64], scalar: f64) {
        self.ixs.iter().enumerate().for_each(|(ix_pos, ix)| {
            if ix_pos < self.twos_from {
                lhs[*ix as usize] += scalar
            } else {
                lhs[*ix as usize] += scalar + scalar
            }
        });
    }
}

pub struct BedVec {
    ones: Vec<u32>,
    twos: Vec<u32>,
}

impl BedVec {
    pub fn new(genotypes: &[u8]) -> Self {
        let mut ones = Vec::new();
        let mut twos = Vec::new();
        for (ix, g) in genotypes.iter().enumerate() {
            match g {
                1 => ones.push(ix as u32),
                2 => twos.push(ix as u32),
                _ => continue,
            }
        }

        Self { ones, twos }
    }

    pub fn scaled_add(&self, lhs: &mut [f64], scalar: f64) {
        self.ones.iter().for_each(|ix| lhs[*ix as usize] += scalar);
        self.twos
            .iter()
            .for_each(|ix| lhs[*ix as usize] += scalar + scalar);
    }
}

pub fn random_genotype_vec(n: usize, maf: f64) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let bin = Binomial::new(maf, 2).unwrap();
    (0..n)
        .map(|_| bin.sample(&mut rng) as u8)
        .collect::<Vec<u8>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bin_bed_bev_rm_mul_with_vec() {
        let num_individuals = 2;
        let num_markers = 4;
        let data: Vec<u8> = vec![0b11000110, 0b10010011];
        let v: Vec<f32> = vec![1., 2., 3., 4.];
        let x = BinBedVecRM::new(data, num_individuals, num_markers);
        assert_eq!(vec![7., 8.], x.mul_with_vec(&v));
    }

    #[test]
    fn test_scaled_add() {
        let n = 100;
        let maf = 0.2;
        let w = 0.6;
        let gtv = random_genotype_vec(n, maf);
        let expected = gtv.iter().map(|e| *e as f64 * w).collect::<Vec<f64>>();
        let bv = BedVec::new(&gtv);
        let mut res_v = vec![0.; n];
        bv.scaled_add(&mut res_v, w);
        assert_eq!(res_v, expected);
    }

    #[test]
    fn test_scaled_add_contig() {
        let n = 100;
        let maf = 0.2;
        let w = 0.6;
        let gtv = random_genotype_vec(n, maf);
        let expected = gtv.iter().map(|e| *e as f64 * w).collect::<Vec<f64>>();
        let bvc = BedVecContig::new(&gtv);
        let mut res_v = vec![0.; n];
        bvc.scaled_add(&mut res_v, w);
        assert_eq!(res_v, expected);
    }
}

#[cfg(all(feature = "unstable", test))]
mod bench {
    extern crate test;

    use super::*;
    use ndarray::arr1;
    use test::Bencher;

    fn prep() -> (BedVec, BedVecContig, Vec<u8>) {
        let n = 1000;
        let maf = 0.2;
        let gtv = random_genotype_vec(n, maf);
        let bv = BedVec::new(&gtv);
        let bvc = BedVecContig::new(&gtv);
        (bv, bvc, gtv)
    }

    #[bench]
    fn bench_scalar_add(b: &mut Bencher) {
        let w = 0.6;
        let (bv, _bvc, gtv) = prep();
        let mut res_v = vec![0.; gtv.len()];
        b.iter(|| {
            bv.scaled_add(&mut res_v, w);
        });
    }

    #[bench]
    fn bench_scalar_add_contig(b: &mut Bencher) {
        let w = 0.6;
        let (_bv, bvc, gtv) = prep();
        let mut res_v = vec![0.; gtv.len()];
        b.iter(|| {
            bvc.scaled_add(&mut res_v, w);
        });
    }

    #[bench]
    fn bench_naive_add(b: &mut Bencher) {
        let w = 0.6;
        let (_bv, _bvc, gtv) = prep();
        let mut res_v = vec![0.; gtv.len()];
        b.iter(|| {
            res_v = gtv.iter().map(|e| *e as f64 * w).collect::<Vec<f64>>();
        });
    }
}
