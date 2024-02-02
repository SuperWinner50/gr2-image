use palette::{cast::from_component_slice};
use palette::{IntoColor, Lab, Srgb};
use kmeans_colors::get_kmeans_hamerly;
use rayon::prelude::*;

const KMEANS_RUNS: usize = 8;

pub fn get_kmeans_colormap(data: Vec<f32>, k: usize) -> (Vec<Lab>, Vec<u8>) {
    let lab: Vec<Lab> = from_component_slice::<Srgb<f32>>(&data)
        .iter()
        .map(|x| x.into_format().into_color())
        .collect();

    let seed = rand::random::<u32>() as u64;
    let result = (0..KMEANS_RUNS).into_par_iter()
        .map(|i| {
            get_kmeans_hamerly(
                k,
                50,
                0.5,
                true,
                &lab,
                seed + i as u64,
            )
        })
        .reduce_with(|a, b| if a.score < b.score { a } else { b })
        .unwrap();

    println!("Kmeans min score: {}", result.score);

    (result.centroids, result.indices)
}