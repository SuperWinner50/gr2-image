use palette::{cast::from_component_slice};
use palette::{IntoColor, Lab, Srgb};
use palette::color_difference::ImprovedCiede2000;
use rayon::prelude::*;

const KMEANS_RUNS: usize = 8;

pub fn get_kmeans_colormap(data: Vec<f32>, k: usize) -> (Vec<Lab>, Vec<u8>) {
    let lab: Vec<Lab> = from_component_slice::<Srgb<f32>>(&data)
        .iter()
        .map(|x| x.into_format().into_color())
        .collect();

    let seed = 2;
    let s = std::time::Instant::now();
    let result = (0..KMEANS_RUNS).into_par_iter()
        .map(|i| {
            kmeans_colors::get_kmeans_hamerly(
                k,
                50,
                1.0,
                true,
                &lab,
                seed + i as u64,
            )
        })
        .reduce_with(|a, b| if a.score < b.score { a } else { b })
        .unwrap();

    let e = s.elapsed();

    let colors = (0..result.indices.len()).map(|i| result.centroids[result.indices[i] as usize]).collect::<Vec<_>>();
    println!("Distance: {}", get_diff_lab(&colors, &lab));
    println!("Kmeans min score: {}", result.score);
    println!("Time: {e:?}");

    (result.centroids, result.indices)
}

fn get_diff_lab(im1: &[Lab], im2: &[Lab]) -> f32 {
    assert_eq!(im1.len(), im2.len());

    let mut dist = 0.0;
    for i in 0..im1.len() {
        let c1: Lab = im1[i];
        let c2: Lab = im2[i];
        dist += c1.improved_difference(c2);
    }

    dist
}
