use crate::bed_lookup_tables::*;
use crate::simd::*;
use ndarray::Array1;
use rayon::prelude::*;

/// Colum-major bed-data in memory.
pub struct BedVecCM {
    data: Vec<u8>,
    col_means: Vec<f32>,
    col_std: Vec<f32>,
    num_individuals: usize,
    num_markers: usize,
    bytes_per_col: usize,
}

impl BedVecCM {
    pub fn new(data: Vec<u8>, num_individuals: usize, num_markers: usize) -> Self {
        let bytes_per_col = if (num_individuals % 4) == 0 {
            num_individuals / 4
        } else {
            num_individuals / 4 + 1
        };
        let mut res = Self {
            data,
            col_means: vec![0.; num_markers],
            col_std: vec![0.; num_markers],
            num_individuals,
            num_markers,
            bytes_per_col,
        };
        res.compute_col_stats();
        res
    }

    pub fn num_individuals(&self) -> usize {
        self.num_individuals
    }

    pub fn num_markers(&self) -> usize {
        self.num_markers
    }

    pub fn data(&self) -> Vec<u8> {
        self.data.clone()
    }

    fn compute_col_stats(&mut self) {
        let mut n: Vec<f32> = vec![0.; self.num_markers];
        for (ix, byte) in self.data.iter().enumerate() {
            let col_ix = (ix * 4) / self.num_individuals;
            let unpacked_byte = unpack_byte_to_genotype_and_validity(byte);
            let mean_update = unpacked_byte[0] * unpacked_byte[4]
                + unpacked_byte[1] * unpacked_byte[5]
                + unpacked_byte[2] * unpacked_byte[6]
                + unpacked_byte[3] * unpacked_byte[7];
            let n_update =
                unpacked_byte[4] + unpacked_byte[5] + unpacked_byte[6] + unpacked_byte[7];
            self.col_means[col_ix] += mean_update;
            n[col_ix] += n_update;
        }
        for (ix, e) in self.col_means.iter_mut().enumerate() {
            *e /= n[ix];
        }
        for (ix, byte) in self.data.iter().enumerate() {
            let col_ix = (ix * 4) / self.num_individuals;
            let unpacked_byte = unpack_byte_to_genotype_and_validity(byte);
            let std_update = ((unpacked_byte[0] - self.col_means[col_ix]) * unpacked_byte[4])
                .powf(2.)
                + ((unpacked_byte[1] - self.col_means[col_ix]) * unpacked_byte[5]).powf(2.)
                + ((unpacked_byte[2] - self.col_means[col_ix]) * unpacked_byte[6]).powf(2.)
                + ((unpacked_byte[3] - self.col_means[col_ix]) * unpacked_byte[7]).powf(2.);
            self.col_std[col_ix] += std_update;
        }
        for (ix, e) in self.col_std.iter_mut().enumerate() {
            *e = (*e / (n[ix] - 1.)).sqrt();
        }
    }

    pub fn left_multiply_seq(&self, v: &[f32]) -> Vec<f32> {
        (0..self.num_markers)
            .map(|col_ix| self.col_dot_product_seq(col_ix, v))
            .collect()
    }

    pub fn left_multiply_simd_v1_seq(&self, v: &[f32]) -> Vec<f32> {
        (0..self.num_markers)
            .map(|col_ix| self.col_dot_product_simd_v1_seq(col_ix, v))
            .collect()
    }

    pub fn left_multiply_par(&self, v: &[f32]) -> Vec<f32> {
        (0..self.num_markers)
            .into_par_iter()
            .map(|col_ix| self.col_dot_product_par(col_ix, v))
            .collect()
    }

    pub fn left_multiply_simd_v1_par(&self, v: &[f32]) -> Array1<f32> {
        Array1::from_vec(
            (0..self.num_markers)
                .into_par_iter()
                .map(|col_ix| self.col_dot_product_simd_v1_par(col_ix, v))
                .collect(),
        )
    }

    pub fn right_multiply_par(&self, v: &[f32]) -> Array1<f32> {
        (0..self.num_markers)
            .into_par_iter()
            .fold(
                // TODO: num_individuals here is problematic if num_individuals is not divisible by four.
                // then the indexing in the last byte will be out of bounds.
                || Array1::zeros(self.num_individuals),
                |mut res, col_ix| {
                    let start_ix = col_ix * self.bytes_per_col;
                    for (byte_ix, byte) in self.data[start_ix..start_ix + self.bytes_per_col]
                        .iter()
                        .enumerate()
                    {
                        let row_ix = byte_ix * 4;
                        let unpacked_byte = unpack_byte_to_genotype_and_validity(byte);
                        res[row_ix] += (unpacked_byte[0] - self.col_means[col_ix])
                            / self.col_std[col_ix]
                            * unpacked_byte[4]
                            * v[col_ix];
                        res[row_ix + 1] += (unpacked_byte[1] - self.col_means[col_ix])
                            / self.col_std[col_ix]
                            * unpacked_byte[5]
                            * v[col_ix];
                        res[row_ix + 2] += (unpacked_byte[2] - self.col_means[col_ix])
                            / self.col_std[col_ix]
                            * unpacked_byte[6]
                            * v[col_ix];
                        res[row_ix + 3] += (unpacked_byte[3] - self.col_means[col_ix])
                            / self.col_std[col_ix]
                            * unpacked_byte[7]
                            * v[col_ix];
                    }
                    res
                },
            )
            .reduce(
                || Array1::zeros(self.num_individuals),
                |mut res, v| {
                    (0..res.len()).for_each(|ix| res[ix] += v[ix]);
                    res
                },
            )
    }

    #[inline(never)]
    pub fn col_dot_product_seq(&self, col_ix: usize, v: &[f32]) -> f32 {
        let start_ix = col_ix * self.bytes_per_col;
        let (xy_sum, y_sum) = self.data[start_ix..start_ix + self.bytes_per_col]
            .iter()
            .enumerate()
            .fold((0., 0.), |(xy_sum, y_sum), (byte_ix, byte)| {
                let unpacked_byte = unpack_byte_to_genotype_and_validity(byte);
                let v_ix = byte_ix * 4;
                // this can crash if in the last byte and v is not padded with 0s
                (
                    xy_sum
                        + unpacked_byte[0] * v[v_ix]
                        + unpacked_byte[1] * v[v_ix + 1]
                        + unpacked_byte[2] * v[v_ix + 2]
                        + unpacked_byte[3] * v[v_ix + 3],
                    y_sum
                        + unpacked_byte[4] * v[v_ix]
                        + unpacked_byte[5] * v[v_ix + 1]
                        + unpacked_byte[6] * v[v_ix + 2]
                        + unpacked_byte[7] * v[v_ix + 3],
                )
            });
        (xy_sum - self.col_means[col_ix] * y_sum) / self.col_std[col_ix]
    }

    #[inline(never)]
    pub fn col_dot_product_simd_v1_seq(&self, col_ix: usize, v: &[f32]) -> f32 {
        let start_ix = col_ix * self.bytes_per_col;
        let xysum = self.data[start_ix..start_ix + self.bytes_per_col]
            .iter()
            .enumerate()
            .fold(f32x8_from_slice(&[0.0_f32; 8]), |xysum, (byte_ix, byte)| {
                let unpacked_byte = unpack_byte_to_genotype_and_validity_f32x8(byte);
                let v_ix = byte_ix * 4;
                let weights = broadcast_into_f32x8(&v[v_ix..v_ix + 4]);
                // this can crash if in the last byte and v is not padded with 0s
                add_f32x8(xysum, multiply_f32x8(unpacked_byte, weights))
            });
        // TODO: check that the indices check out
        ((extract(xysum, 0) + extract(xysum, 1) + extract(xysum, 2) + extract(xysum, 3))
            - self.col_means[col_ix]
                * (extract(xysum, 4) + extract(xysum, 5) + extract(xysum, 6) + extract(xysum, 7)))
            / self.col_std[col_ix]
    }

    // try using 8 f32s at a time, I have enough ymm registers for this.
    // in order to avoid using a remainder, it would be useful to have bytes_per_col be divisible by 2.
    // if it isn't, I can add a padding byte full of nan.
    #[inline(never)]
    pub fn col_dot_product_simd_v2_seq(&self, _col_ix: usize, _v: &[f32]) -> f32 {
        unimplemented!();
        // let start_ix = col_ix * self.bytes_per_col;
        // let xysum = self.data[start_ix..start_ix + self.bytes_per_col]
        //     .chunks_exact(2)
        //     .enumerate()
        //     .fold(from_slice(&[0.0_f32; 8]), |xysum, (byte_ix, byte)| {
        //         let unpacked_byte = unpack_byte_to_genotype_and_validity_f32x8(byte);
        //         let v_ix = byte_ix * 4;
        //         let weights = broadcast_into_f32x8(&v[v_ix..v_ix + 4]);
        //         // this can crash if in the last byte and v is not padded with 0s
        //         add(xysum, multiply(unpacked_byte, weights))
        //     });
        // // TODO: check that the indices check out
        // ((extract(xysum, 0) + extract(xysum, 1) + extract(xysum, 2) + extract(xysum, 3))
        //     - self.col_means[col_ix]
        //         * (extract(xysum, 4) + extract(xysum, 5) + extract(xysum, 6) + extract(xysum, 7)))
        //     / self.col_std[col_ix]
    }

    #[inline(always)]
    pub fn col_dot_product_par(&self, col_ix: usize, v: &[f32]) -> f32 {
        let start_ix = col_ix * self.bytes_per_col;
        let (xy_sum, y_sum) = self.data[start_ix..start_ix + self.bytes_per_col]
            .par_iter()
            .enumerate()
            .fold(
                || (0., 0.),
                |(xy_sum, y_sum), (byte_ix, byte)| {
                    let unpacked_byte = unpack_byte_to_genotype_and_validity(byte);
                    let v_ix = byte_ix * 4;
                    // this can crash if in the last byte and v is not padded with 0s
                    (
                        xy_sum
                            + unpacked_byte[0] * v[v_ix]
                            + unpacked_byte[1] * v[v_ix + 1]
                            + unpacked_byte[2] * v[v_ix + 2]
                            + unpacked_byte[3] * v[v_ix + 3],
                        y_sum
                            + unpacked_byte[4] * v[v_ix]
                            + unpacked_byte[5] * v[v_ix + 1]
                            + unpacked_byte[6] * v[v_ix + 2]
                            + unpacked_byte[7] * v[v_ix + 3],
                    )
                },
            )
            .reduce(
                || (0., 0.),
                |sum, next_summand| (sum.0 + next_summand.0, sum.1 + next_summand.1),
            );
        (xy_sum - self.col_means[col_ix] * y_sum) / self.col_std[col_ix]
    }

    #[inline(always)]
    pub fn col_dot_product_simd_v1_par(&self, col_ix: usize, v: &[f32]) -> f32 {
        let start_ix = col_ix * self.bytes_per_col;
        let xysum = self.data[start_ix..start_ix + self.bytes_per_col]
            .par_iter()
            .enumerate()
            .fold(
                || f32x8_from_slice(&[0.0_f32; 8]),
                |xysum, (byte_ix, byte)| {
                    let unpacked_byte = unpack_byte_to_genotype_and_validity_f32x8(byte);
                    let v_ix = byte_ix * 4;
                    let weights = broadcast_into_f32x8(&v[v_ix..v_ix + 4]);
                    // this can crash if in the last byte and v is not padded with 0s
                    add_f32x8(xysum, multiply_f32x8(unpacked_byte, weights))
                },
            )
            .reduce(|| f32x8_from_slice(&[0.0_f32; 8]), add_f32x8);
        // TODO: check that the indices check out
        ((extract(xysum, 0) + extract(xysum, 1) + extract(xysum, 2) + extract(xysum, 3))
            - self.col_means[col_ix]
                * (extract(xysum, 4) + extract(xysum, 5) + extract(xysum, 6) + extract(xysum, 7)))
            / self.col_std[col_ix]
    }
}

#[inline(always)]
fn unpack_byte_to_genotype_and_validity(byte: &u8) -> [f32; 8] {
    let start_ix = *byte as usize * 8;
    BED_LOOKUP_GENOTYPE_AND_VALIDITY[start_ix..start_ix + 8]
        .try_into()
        .expect("Failed to unpack bed byte")
}

#[inline(always)]
fn unpack_byte_to_genotype_and_validity_f32x8(byte: &u8) -> f32x8 {
    let start_ix = *byte as usize * 8;
    f32x8_from_slice(
        BED_LOOKUP_GENOTYPE_AND_VALIDITY[start_ix..start_ix + 8]
            .try_into()
            .expect("Failed to unpack bed byte"),
    )
}

#[inline(always)]
fn unpack_byte_to_genotype_f32x4(byte: &u8) -> f32x4 {
    let start_ix = *byte as usize * 4;
    f32x4_from_slice(
        BED_LOOKUP_GENOTYPE[start_ix..start_ix + 4]
            .try_into()
            .expect("Failed to unpack bed byte"),
    )
}

#[inline(always)]
fn unpack_byte_to_validity_f32x4(byte: &u8) -> f32x4 {
    let start_ix = *byte as usize * 4;
    f32x4_from_slice(
        BED_LOOKUP_VALIDITY[start_ix..start_ix + 4]
            .try_into()
            .expect("Failed to unpack bed byte"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_bed_vec_cm_stats() {
        let num_individuals = 4;
        let num_markers = 4;
        let data: Vec<u8> = vec![0b01001011, 0b11101101, 0b11111110, 0b10110011];
        let x = BedVecCM::new(data, num_individuals, num_markers);
        // m.T = (
        //  0., 1., 2., na,
        //  na, 0., 1., 0.,
        //  1., 0., 0., 0.,
        //  0., 2., 0., 1.,
        // )
        // m = (
        //  0., na, 1., 0.,
        //  1., 0., 0., 2.,
        //  2., 1., 0., 0.,
        //  na, 0., 0., 1.,
        // )
        assert_eq!(x.col_means, vec![1., (1. / 3.), 0.25, 0.75]);
        assert_eq!(x.col_std, vec![1.0, 0.57735026, 0.5, 0.95742714]);
    }

    #[test]
    fn test_bed_vec_cm_left_multiply() {
        let num_individuals = 4;
        let num_markers = 4;
        let data: Vec<u8> = vec![0b01001011, 0b11101101, 0b11111110, 0b10110011];
        let x = BedVecCM::new(data, num_individuals, num_markers);
        let v: Vec<f32> = vec![1., 1., 1., 1.];
        assert_eq!(vec![0., 0., 0., 0.], x.left_multiply_seq(&v));
        assert_eq!(vec![0., 0., 0., 0.], x.left_multiply_par(&v));
        let v: Vec<f32> = vec![2., 0., -1., 4.];
        assert_eq!(
            vec![-3.0, -3.4641018, 1.5, 0.26111647],
            x.left_multiply_seq(&v)
        );
        assert_eq!(
            vec![-3.0, -3.4641018, 1.5, 0.26111647],
            x.left_multiply_par(&v)
        );
    }

    #[test]
    fn test_bed_vec_cm_left_multiply_simd_v1() {
        let num_individuals = 4;
        let num_markers = 4;
        let data: Vec<u8> = vec![0b01001011, 0b11101101, 0b11111110, 0b10110011];
        let x = BedVecCM::new(data, num_individuals, num_markers);
        let v: Vec<f32> = vec![1., 1., 1., 1.];
        assert_eq!(vec![0., 0., 0., 0.], x.left_multiply_simd_v1_seq(&v));
        assert_eq!(arr1(&[0., 0., 0., 0.]), x.left_multiply_simd_v1_par(&v));
        let v: Vec<f32> = vec![2., 0., -1., 4.];
        assert_eq!(
            vec![-3.0, -3.4641018, 1.5, 0.26111647],
            x.left_multiply_simd_v1_seq(&v)
        );
        assert_eq!(
            arr1(&[-3.0, -3.4641018, 1.5, 0.26111647]),
            x.left_multiply_simd_v1_par(&v)
        );
    }

    #[test]
    fn test_bed_vec_cm_right_multiply() {
        let num_individuals = 4;
        let num_markers = 4;
        let data: Vec<u8> = vec![0b01001011, 0b11101101, 0b11111110, 0b10110011];
        let x = BedVecCM::new(data, num_individuals, num_markers);
        let v = vec![1., 1., 1., 1.];
        assert_eq!(
            arr1(&[-0.2833494, 0.22823209, 0.8713511, -0.8162339]),
            x.right_multiply_par(&v)
        );
    }
}
