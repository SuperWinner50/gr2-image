use palette::{cast::from_component_slice};
use palette::{IntoColor, FromColor, Lab, Srgb};
use palette::color_difference::ImprovedCiede2000;
use rayon::prelude::*;

use quantette::{UnsizedPipeline, ColorSpace, QuantizeMethod, kmeans::Centroids, KmeansOptions};

const KMEANS_RUNS: usize = 30;

pub fn get_kmeans_colormap(data: Vec<f32>, k: usize, init: Vec<Lab>) -> (Vec<Lab>, Vec<u8>) {
    let lab: Vec<Lab> = from_component_slice::<Srgb<f32>>(&data)
        .iter()
        .map(|x| x.into_format().into_color())
        .collect();

    let seed = 2;
    let s = std::time::Instant::now();
    let result = (0..KMEANS_RUNS).into_par_iter()
        .map(|i| {
            get_kmeans_hamerly(
                k,
                200,
                1.0,
                true,
                &lab,
                seed + i as u64,
                &init,
            )
        })
        .reduce_with(|a, b| if a.score < b.score { a } else { b })
        .unwrap();

    let e = s.elapsed();

    let colors = (0..result.indices.len()).map(|i| result.centroids[result.indices[i] as usize]).collect::<Vec<_>>();
    // let colors2 = lab.into_iter().map(|v| <palette::Lab as IntoColor<Srgb<f32>>>::into_color(v).into_format()).collect();
    println!("Distance: {}", get_diff_lab(&colors, &lab));
    println!("Kmeans min score: {}", result.score);
    println!("Time: {e:?}");

    (result.centroids, result.indices)
}

fn get_diff(im1: &[Srgb<u8>], im2: &[Srgb<u8>]) -> f32 {
    assert_eq!(im1.len(), im2.len());

    let mut dist = 0.0;
    for i in 0..im1.len() {
        let c1: Lab = im1[i].into_format().into_color();
        let c2: Lab = im2[i].into_format().into_color();
        dist += c1.improved_difference(c2);
    }

    dist
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

pub fn get_kmeans_colormap_o(data: Vec<f32>, k: u8) -> (Vec<Lab>, Vec<u8>) {    
    let colors = from_component_slice::<Srgb<f32>>(data.as_slice())
        .into_iter()
        .map(|c| c.into_format())
        .collect::<Vec<_>>();
    let slice = colors.as_slice().try_into().unwrap();

    let options = KmeansOptions::new()
        .sampling_factor(1.0)
        .batch_size(128)
        .seed(1);

    let s = std::time::Instant::now();
    let cmap: Vec<Lab> = UnsizedPipeline::new(slice)
        .palette_size(k.into())
        .colorspace(ColorSpace::Lab)
        .dedup_pixels(false)
        .quantize_method(QuantizeMethod::Kmeans(options))
        .palette()
        .into_iter()
        .map(|c| c.into_format().into_color())
        .collect();

    let (im, inds) = get_kmeans_colormap(data, k as usize, cmap);

    let e = s.elapsed();

    // println!("Kmeans min score: {}", result.score);
    println!("Time: {e:?}");
    let new_colors = (0..inds.len()).map(|i| im[inds[i] as usize]).collect::<Vec<_>>();
    // println!("Distance: {}", get_diff_lab(&new_colors, &colors));
    (im, inds)
    // println!("Palette: {result:?}");

    // (result.centroids, result.indices)
}

use kmeans_colors::{Kmeans, HamerlyPoint, HamerlyCentroids, Calculate, Hamerly};
use rand::SeedableRng;

pub fn get_kmeans_hamerly<C: Hamerly + Clone>(
    k: usize,
    max_iter: usize,
    converge: f32,
    verbose: bool,
    buf: &[C],
    seed: u64,
    initial_centroids: &[C],
) -> Kmeans<C> {
    // Initialize the random centroids
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut centers: HamerlyCentroids<C> = HamerlyCentroids::new(k);
    kmeans_colors::init_plus_plus(k, &mut rng, buf, &mut centers.centroids);

    // Initialize points buffer and convergence variables
    let mut iterations = 0;
    let mut score;
    let mut old_centers = centers.centroids.clone();
    let mut points: Vec<HamerlyPoint> = (0..buf.len()).map(|_| HamerlyPoint::new()).collect();

    // Main loop: find nearest centroids and recalculate means until convergence
    loop {
        C::compute_half_distances(&mut centers);
        C::get_closest_centroid_hamerly(buf, &centers, &mut points);
        C::recalculate_centroids_hamerly(&mut rng, buf, &mut centers, &points);

        score = Calculate::check_loop(&centers.centroids, &old_centers);
        if verbose {
            println!("Score: {}", score);
        }

        // Verify that either the maximum iteration count has been met or the
        // centroids haven't moved beyond a certain threshold since the
        // previous iteration.
        if iterations >= max_iter || score <= converge {
            if verbose {
                println!("Iterations: {}", iterations);
            }
            break;
        }

        C::update_bounds(&centers, &mut points);
        old_centers.clone_from(&centers.centroids);
        iterations += 1;
    }

    Kmeans {
        score,
        centroids: centers.centroids,
        indices: points.iter().map(|x| x.index).collect(),
    }
}