fn main() {
    println!("//! Lookup tables for unpacking of bed bytes");
    println!("");
    // genotype value lookups
    println!("/// Lookup table for unpacking of genotype values from bed bytes");
    println!("pub const BED_LOOKUP_GENOTYPE: [f32; 1024] = [");
    (0..=255).for_each(|v| {
        let mut b: u8 = v;
        for _ in 0..4 {
            match b & 3 {
                0 => println!("\t2.,"),
                1 => println!("\t0.,"),
                2 => println!("\t1.,"),
                3 => println!("\t0.,"),
                _ => panic!("unexpected bit value"),
            }
            b >>= 2;
        }
    });
    println!("];");
    println!("");
    // validity lookups
    println!("/// Lookup table for marking valid vs missing data in bed bytes");
    println!("pub const BED_LOOKUP_VALIDITY: [f32; 1024] = [");
    (0..=255).for_each(|v| {
        let mut b: u8 = v;
        for _ in 0..4 {
            match b & 3 {
                0 => println!("\t1.,"),
                1 => println!("\t0.,"),
                2 => println!("\t1.,"),
                3 => println!("\t1.,"),
                _ => panic!("unexpected bit value"),
            }
            b >>= 2;
        }
    });
    println!("];");
    println!("");
    // combined lookups
    println!("/// Lookup table for genotype values and valid data markers");
    println!("pub const BED_LOOKUP_GENOTYPE_AND_VALIDITY: [f32; 2048] = [");
    (0..=255).for_each(|v| {
        let mut b: u8 = v;
        for _ in 0..4 {
            match b & 3 {
                0 => println!("\t2.,"),
                1 => println!("\t0.,"),
                2 => println!("\t1.,"),
                3 => println!("\t0.,"),
                _ => panic!("unexpected bit value"),
            }
            b >>= 2;
        }
        let mut b: u8 = v;
        for _ in 0..4 {
            match b & 3 {
                0 => println!("\t1.,"),
                1 => println!("\t0.,"),
                2 => println!("\t1.,"),
                3 => println!("\t1.,"),
                _ => panic!("unexpected bit value"),
            }
            b >>= 2;
        }
    });
    println!("];");
    println!("");
}
