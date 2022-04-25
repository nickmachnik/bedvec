use crate::bed_lookup_tables::*;
use rand::Rng;
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
        self.num_individuals()
    }

    fn compute_col_stats(&mut self) {
        let mut n: Vec<f32> = vec![0.; self.num_markers];
        for (ix, byte) in self.data.iter().enumerate() {
            let col_ix = (ix * 4) / self.num_individuals;
            let unpacked_byte = self.unpack_byte_to_genotype_and_validity(byte);
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
            let unpacked_byte = self.unpack_byte_to_genotype_and_validity(byte);
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

    pub fn left_multiply(&self, v: &[f32]) -> Vec<f32> {
        (0..self.num_markers)
            .into_par_iter()
            .map(|col_ix| self.col_dot_product_map_reduce(col_ix, v))
            .collect()
    }

    #[inline(always)]
    pub fn col_dot_product_fold_reduce(&self, col_ix: usize, v: &[f32]) -> f32 {
        let start_ix = col_ix * self.bytes_per_col;
        let (xy_sum, y_sum) = self.data[start_ix..start_ix + self.bytes_per_col]
            .par_iter()
            .enumerate()
            .fold(
                || (0., 0.),
                |(xy_sum, y_sum), (byte_ix, byte)| {
                    let unpacked_byte = self.unpack_byte_to_genotype_and_validity(byte);
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
    pub fn col_dot_product_map_reduce(&self, col_ix: usize, v: &[f32]) -> f32 {
        let start_ix = col_ix * self.bytes_per_col;
        let (xy_sum, y_sum) = self.data[start_ix..start_ix + self.bytes_per_col]
            .par_iter()
            .enumerate()
            .map(|(byte_ix, byte)| {
                let unpacked_byte = self.unpack_byte_to_genotype_and_validity(byte);
                let v_ix = byte_ix * 4;
                // this can crash if in the last byte and v is not padded with 0s
                (
                    unpacked_byte[0] * v[v_ix]
                        + unpacked_byte[1] * v[v_ix + 1]
                        + unpacked_byte[2] * v[v_ix + 2]
                        + unpacked_byte[3] * v[v_ix + 3],
                    unpacked_byte[4] * v[v_ix]
                        + unpacked_byte[5] * v[v_ix + 1]
                        + unpacked_byte[6] * v[v_ix + 2]
                        + unpacked_byte[7] * v[v_ix + 3],
                )
            })
            .reduce(
                || (0., 0.),
                |sum, next_summand| (sum.0 + next_summand.0, sum.1 + next_summand.1),
            );
        (xy_sum - self.col_means[col_ix] * y_sum) / self.col_std[col_ix]
    }

    #[inline(always)]
    fn unpack_byte_to_genotype_and_validity(&self, byte: &u8) -> [f32; 8] {
        let start_ix = *byte as usize * 8;
        BED_LOOKUP_GENOTYPE_AND_VALIDITY[start_ix..start_ix + 8]
            .try_into()
            .expect("Failed to unpack bed byte")
    }
}

/// Row-major bed-data in memory.
pub struct BedVecRM {
    data: Vec<u8>,
    col_means: Vec<f32>,
    col_std: Vec<f32>,
    num_individuals: usize,
    num_markers: usize,
    // row_padding_bits: usize,
    bytes_per_row: usize,
}

impl BedVecRM {
    pub fn new(data: Vec<u8>, num_individuals: usize, num_markers: usize) -> Self {
        let bytes_per_row = if (num_markers % 4) == 0 {
            num_markers / 4
        } else {
            num_markers / 4 + 1
        };
        let mut res = Self {
            data,
            col_means: vec![0.; num_markers],
            col_std: vec![0.; num_markers],
            num_individuals,
            num_markers,
            bytes_per_row,
        };
        res.compute_col_stats();
        res
    }

    /// Create a completely random new BedVecRM of given dimensions.
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
        let mut res = Self {
            data,
            col_means: vec![0.; num_markers],
            col_std: vec![0.; num_markers],
            num_individuals,
            num_markers,
            bytes_per_row,
        };
        res.compute_col_stats();
        res
    }

    fn compute_col_stats(&mut self) {
        let mut n: Vec<f32> = vec![0.; self.num_markers];
        for (ix, byte) in self.data.iter().enumerate() {
            let byte_start_col_ix = (ix * 4) % (self.bytes_per_row * 4);
            let unpacked_byte = self.unpack_byte_to_genotype_and_validity(byte);
            self.col_means[byte_start_col_ix] += unpacked_byte[0] * unpacked_byte[4];
            n[byte_start_col_ix] += unpacked_byte[4];
            self.col_means[byte_start_col_ix + 1] += unpacked_byte[1] * unpacked_byte[5];
            n[byte_start_col_ix + 1] += unpacked_byte[5];
            self.col_means[byte_start_col_ix + 2] += unpacked_byte[2] * unpacked_byte[6];
            n[byte_start_col_ix + 2] += unpacked_byte[6];
            self.col_means[byte_start_col_ix + 3] += unpacked_byte[3] * unpacked_byte[7];
            n[byte_start_col_ix + 3] += unpacked_byte[7];
        }
        for (ix, e) in self.col_means.iter_mut().enumerate() {
            *e /= n[ix];
        }
        for (ix, byte) in self.data.iter().enumerate() {
            let byte_start_col_ix = (ix * 4) % (self.bytes_per_row * 4);
            let unpacked_byte = self.unpack_byte_to_genotype_and_validity(byte);
            self.col_std[byte_start_col_ix] +=
                ((unpacked_byte[0] - self.col_means[byte_start_col_ix]) * unpacked_byte[4])
                    .powf(2.);
            self.col_std[byte_start_col_ix + 1] +=
                ((unpacked_byte[1] - self.col_means[byte_start_col_ix + 1]) * unpacked_byte[5])
                    .powf(2.);
            self.col_std[byte_start_col_ix + 2] +=
                ((unpacked_byte[2] - self.col_means[byte_start_col_ix + 2]) * unpacked_byte[6])
                    .powf(2.);
            self.col_std[byte_start_col_ix + 3] +=
                ((unpacked_byte[3] - self.col_means[byte_start_col_ix + 3]) * unpacked_byte[7])
                    .powf(2.);
        }
        for (ix, e) in self.col_std.iter_mut().enumerate() {
            *e = (*e / (n[ix] - 1.)).sqrt();
        }
    }

    pub fn right_multiply(&self, v: &[f32]) -> Vec<f32> {
        (0..self.num_individuals)
            .into_par_iter()
            .map(|row_ix| self.row_dot_product(row_ix, v))
            .collect()
    }

    #[inline(always)]
    fn row_dot_product(&self, row_ix: usize, v: &[f32]) -> f32 {
        let start_ix = row_ix * self.bytes_per_row;
        self.data[start_ix..start_ix + self.bytes_per_row]
            .par_iter()
            .enumerate()
            .map(|(byte_ix, byte)| {
                let unpacked_byte = self.unpack_byte_to_genotype_and_validity(byte);
                let v_ix = byte_ix * 4;
                // this is (x_j - mu_j * I[x_j is not na]) / sig_j * v_j
                ((unpacked_byte[0] - self.col_means[v_ix]) * unpacked_byte[4] / self.col_std[v_ix]
                    * v[v_ix])
                    + ((unpacked_byte[1] - self.col_means[v_ix + 1]) * unpacked_byte[5]
                        / self.col_std[v_ix + 1]
                        * v[v_ix + 1])
                    + ((unpacked_byte[2] - self.col_means[v_ix + 2]) * unpacked_byte[6]
                        / self.col_std[v_ix + 2]
                        * v[v_ix + 2])
                    + ((unpacked_byte[3] - self.col_means[v_ix + 3]) * unpacked_byte[7]
                        / self.col_std[v_ix + 3]
                        * v[v_ix + 3])
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bin_vec_rm_stats() {
        let num_individuals = 4;
        let num_markers = 4;
        let data: Vec<u8> = vec![0b01001011, 0b11101101, 0b11111110, 0b10110011];
        let x = BedVecRM::new(data, num_individuals, num_markers);
        // m = (
        //  0., 1., 2., na,
        //  na, 0., 1., 0.,
        //  1., 0., 0., 0.,
        //  0., 2., 0., 1.,
        // )
        assert_eq!(
            x.col_means,
            vec![(1.0_f32 / 3.0_f32), 0.75, 0.75, (1.0_f32 / 3.0_f32)]
        );
        assert_eq!(
            x.col_std,
            vec![
                (1.0_f32 / 3.0_f32).sqrt(),
                (11.0_f32 / 12.0_f32).sqrt(),
                (11.0_f32 / 12.0_f32).sqrt(),
                (1.0_f32 / 3.0_f32).sqrt(),
            ]
        );
    }

    #[test]
    fn test_bin_bed_vec_rm_right_multiply() {
        let num_individuals = 4;
        let num_markers = 4;
        let data: Vec<u8> = vec![0b01001011, 0b11101101, 0b11111110, 0b10110011];
        let v: Vec<f32> = vec![1., 2., 3., 4.];
        let x = BedVecRM::new(data, num_individuals, num_markers);
        assert_eq!(
            vec![3.8616297, -3.0927505, -5.0714474, 4.3025684],
            x.right_multiply(&v)
        );
    }
}
