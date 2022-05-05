//! Functionality for reading .bed data from disk

use crate::bedvec::BedVecCM;

// For now I expect to read whole .bed files in CM order.
// If I go for batching, I will probably still be doing that in memory.
// If not, RM will be necessary for efficient reading of blocks.
pub struct BedReader {
    bed_path: String,
    col_stats_path: Option<String>,
    num_individuals: usize,
    num_markers: usize,
}

impl BedReader {
    pub fn new(bed_path: &str, num_individuals: usize, num_markers: usize) -> Self {
        Self {
            bed_path: bed_path.to_owned(),
            col_stats_path: None,
            num_individuals,
            num_markers,
        }
    }

    pub fn with_col_stats(&mut self, col_stats_path: &str) -> &mut Self {
        self.col_stats_path = Some(col_stats_path.to_owned());
        self
    }

    /// Load bedvec from file.
    ///
    /// This assumes that the bed file starts with info from the first byte on,
    /// i.e. that the typical three byte .bed prefix is removed.
    /// Furthermore, col means and stds are always recomputed, but should
    /// ideally be stored in accompanying files in preprocessing.
    ///
    /// TODO: This could potentially be sped up by memory mapping (memmap) the bed file.
    /// This could use a lot of memory though if all MarkerGroups do this.
    pub fn read_into_bedvec(&self) -> BedVecCM {
        let bytes = std::fs::read(&self.bed_path).expect("failed to read .bed file");
        if let Some(_p) = &self.col_stats_path {
            unimplemented!()
        } else {
            BedVecCM::new(bytes, self.num_individuals, self.num_markers)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bed_loading() {
        let reader = BedReader::new("resources/test/three_by_two.bed", 4, 2);
        let bvcm = reader.read_into_bedvec();
        let exp: Vec<u8> = vec![0xf8, 0x92];
        assert_eq!(bvcm.data(), exp);
    }
}
