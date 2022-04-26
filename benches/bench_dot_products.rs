use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use rs_bedvec::bedvec::BedVecCM;

fn rand_bed_column() -> BedVecCM {
    let mut rng = rand::thread_rng();
    let num_individuals = 10_000;
    let data: Vec<u8> = (0..(num_individuals / 4)).map(|_| rng.gen()).collect();
    BedVecCM::new(data, num_individuals, 1)
}

fn rand_bed_matrix_cm() -> BedVecCM {
    let mut rng = rand::thread_rng();
    let num_individuals = 10_000;
    let num_markers = 1000;
    let data: Vec<u8> = (0..(num_markers * num_individuals / 4))
        .map(|_| rng.gen())
        .collect();
    BedVecCM::new(data, num_individuals, num_markers)
}

fn bench_cm_left_mul_seq(c: &mut Criterion) {
    let mut group = c.benchmark_group("cm_left_mul_seq");
    let bv = rand_bed_matrix_cm();
    let mut rng = rand::thread_rng();
    let left_w: Vec<f32> = (0..bv.num_individuals()).map(|_| rng.gen()).collect();
    for i in [20u64, 21u64].iter() {
        group.bench_with_input(BenchmarkId::new("v0", i), i, |b, i| {
            b.iter(|| bv.left_multiply_seq(&left_w))
        });
        group.bench_with_input(BenchmarkId::new("v1 simd", i), i, |b, i| {
            b.iter(|| bv.left_multiply_simd_v1_seq(&left_w))
        });
    }
    group.finish();
}

fn bench_cm_left_mul_par(c: &mut Criterion) {
    let mut group = c.benchmark_group("cm_left_mul_par");
    let bv = rand_bed_matrix_cm();
    let mut rng = rand::thread_rng();
    let left_w: Vec<f32> = (0..bv.num_individuals()).map(|_| rng.gen()).collect();
    for i in [20u64, 21u64].iter() {
        group.bench_with_input(BenchmarkId::new("v0", i), i, |b, i| {
            b.iter(|| bv.left_multiply_par(&left_w))
        });
        group.bench_with_input(BenchmarkId::new("v1 simd", i), i, |b, i| {
            b.iter(|| bv.left_multiply_simd_v1_par(&left_w))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cm_left_mul_seq, bench_cm_left_mul_par,);

criterion_main!(benches);
