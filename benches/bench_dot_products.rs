use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;
use rs_bedvec::bedvec::BedVecCM;

fn rand_bed_column() -> BedVecCM {
    let mut rng = rand::thread_rng();
    let num_individuals = 10000;
    let data: Vec<u8> = (0..(num_individuals)).map(|_| rng.gen()).collect();
    BedVecCM::new(data, num_individuals, 1)
}

fn bench_bedvec_cm_dot_product_map_reduce(c: &mut Criterion) {
    let bv = rand_bed_column();
    let left_w = vec![1.6; bv.num_individuals()];
    c.bench_function("cm dot map reduce", |b| {
        b.iter(|| bv.col_dot_product_map_reduce(0, &left_w))
    });
}

fn bench_bedvec_cm_dot_product_fold_reduce(c: &mut Criterion) {
    let bv = rand_bed_column();
    let left_w = vec![1.6; bv.num_individuals()];
    c.bench_function("cm dot fold reduce", |b| {
        b.iter(|| bv.col_dot_product_fold_reduce(0, &left_w))
    });
}

criterion_group!(
    benches,
    bench_bedvec_cm_dot_product_map_reduce,
    bench_bedvec_cm_dot_product_fold_reduce
);

criterion_main!(benches);
