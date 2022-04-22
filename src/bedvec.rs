use rand::distributions::Distribution;
use statrs::distribution::Binomial;

pub struct BedVecContig {
    ixs: Vec<u32>,
    twos_from: usize,
}

impl BedVecContig {
    pub fn new(genotypes: &Vec<u8>) -> Self {
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

    pub fn scaled_add(&self, lhs: &mut Vec<f64>, scalar: f64) {
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
    pub fn new(genotypes: &Vec<u8>) -> Self {
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

    pub fn scaled_add(&self, lhs: &mut Vec<f64>, scalar: f64) {
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
    use test::Bencher;
    use ndarray::arr1;

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
