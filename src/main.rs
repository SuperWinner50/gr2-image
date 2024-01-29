use std::collections::HashMap;
use std::io::Write;

use silv::{RadarFile, Sweep, Ray};
use rayon::prelude::*;

use palette::Lab;

fn write_simple_colormap(data: Vec<f32>) -> Vec<u8> {
    let mut writer = std::fs::File::create("colormap.pal").unwrap();

    writeln!(writer, "Product: BR\nUnits:  DBZ\nStep:   10\n").unwrap();
    for c3 in 0..6 {
        for c2 in 0..7 {
            for c1 in 0..6 {
                let idx = (c1 + c2 * 6 + c3 * 6 * 7) as f32 / 2.0 - 32.0;
                let f1 = (c1 as f32 * 255.0 / 5.0).trunc();
                let f2 = (c2 as f32 * 255.0 / 6.0).trunc();
                let f3 = (c3 as f32 * 255.0 / 5.0).trunc();

                writeln!(writer, "Color: {idx} {f1} {f2} {f3} {f1} {f2} {f3}").unwrap();
            }
        }
    }

    data.chunks(3)
        .map(|color| {
            let c1 = 6.0 * clamp(color[0]);
            let c2 = 7.0 * clamp(color[1]);
            let c3 = 6.0 * clamp(color[2]);

            c1 as u8 + c2 as u8 * 6 + c3 as u8 * 6 * 7
        })
        .collect()
}

fn total_cost(colors: &Vec<(usize, Lab)>) -> f32 {
    use palette::color_difference::Ciede2000;    

    colors.windows(2).map(|c| c[0].1.difference(c[1].1).powi(2)).sum::<f32>()
}

fn cost_diff(colors: &Vec<(usize, Lab)>, a: usize, b: usize) -> f32 {
    use palette::color_difference::Ciede2000;

    let mut end = 0.0;
    if a > 0 {
        end += colors[a - 1].1.difference(colors[b].1).powi(2) - colors[a - 1].1.difference(colors[a].1).powi(2);
    }

    if a.abs_diff(b) != 1 {
        if a < colors.len() - 1 {
            end += colors[b].1.difference(colors[a + 1].1).powi(2) - colors[a].1.difference(colors[a + 1].1).powi(2);
        }
        
        if b > 0 {
            end += colors[b - 1].1.difference(colors[a].1).powi(2) - colors[b - 1].1.difference(colors[b].1).powi(2);
        }
    }

    if b < colors.len() - 1 {
        end += colors[a].1.difference(colors[b + 1].1).powi(2) - colors[b].1.difference(colors[b + 1].1).powi(2);
    }

    end
}

fn anneal(colors: &Vec<(usize, Lab)>) -> (Vec<(usize, Lab)>, f32)  {
    use rand::Rng;
    
    let mut colors = colors.clone();

    // Params (experimental)
    let max = 10000.0;
    let a = 15.0;
    let restart_n = ANNEALING_RUNS / 10;

    let mut c = total_cost(&colors);
    let mut best = (0, colors.clone(), c);
    let mut rng = rand::thread_rng();

    for i in 0..ANNEALING_RUNS {
        if i - best.0 > restart_n {
            colors = best.1.clone();
            c = best.2;
            best.0 = i;
        }

        let mut p1 = rng.gen_range(0..colors.len());
        let mut p2 = rng.gen_range(0..colors.len());

        if p1 == p2 {
            continue;
        }

        if p1 > p2 {
            std::mem::swap(&mut p1, &mut p2);
        }

        let temp = max * 0.5f32.powf(a * (i + 1) as f32 / ANNEALING_RUNS as f32) - max * 0.5f32.powf(a);

        let change = cost_diff(&colors, p1, p2);

        if change < temp  {
            c += change;
            colors.swap(p1, p2);
        }

        if c < best.2 {
            best = (i, colors.clone(), c);
        }
    }

    (best.1, best.2)
}

fn write_kmeans_colormap(data: Vec<f32>) -> Vec<u8> {
    use palette::{cast::from_component_slice, color_difference::Ciede2000};
    use palette::{FromColor, IntoColor, Lab, Srgb};
    use kmeans_colors::{get_kmeans_hamerly};
    use rand::seq::SliceRandom;

    let lab: Vec<Lab> = from_component_slice::<Srgb<f32>>(&data)
        .iter()
        .map(|x| x.into_format().into_color())
        .collect();

    // let seed = rand::random::<u32>() as u64;
    let seed = 1;
    let result = (0..KMEANS_RUNS).into_par_iter()
        .map(|i| {
            get_kmeans_hamerly(
                K,
                100,
                0.1,
                true,
                &lab,
                seed + i as u64,
            )
        })
        .reduce_with(|a, b| if a.score < b.score { a } else { b })
        .unwrap();

    println!("Min score: {}", result.score);

    let mut res = result.centroids.clone().into_iter().enumerate().collect::<Vec<_>>();

    let (sorted, cost) = (0..ANNEALING_PAR_N).into_par_iter()
        .map(|_| anneal(&res))
        .reduce_with(|a, b| if a.1 < b.1 { a } else { b })
        .unwrap();

    println!("Coloring cost: {}", cost / K as f32);

    let idxs = sorted.iter().map(|(i, _)| *i).collect::<Vec<_>>();

    let rgb = sorted
        .into_iter()
        .map(|(_, x)| Srgb::from_color(x).into_format())
        .collect::<Vec<Srgb<u8>>>();

    let mut writer = std::fs::File::create("colormap.pal").unwrap();

    writeln!(writer, "Product: BR\nUnits:  DBZ\nStep:   10\n").unwrap();

    for i in 0..K {
        let (r, g, b) = rgb[i].into_components();
        let idx = i as f32 / 2.0 - 32.0;

        writeln!(writer, "Color: {idx} {r} {g} {b} {r} {g} {b}").unwrap();
    }

    let mut idxs2 = (0..K).collect::<Vec<_>>();
    idxs2.sort_by_key(|i| idxs[*i]);

    result.indices.iter().cloned().map(|v| idxs2[v as usize] as u8).collect()
}

fn clamp(x: f32) -> f32 {
    match x {
        x if x <= 0.0 => 0.0001,
        x if x >= 1.0 => 0.9999,
        x => x,
    }
}

#[derive(Clone, Copy)]
enum Fit {
    Max,
    Min,
    Val(f32), 
}

fn weight(image: &image::Rgb32FImage, points: Vec<(f32, f32)>) -> Option<[f32; 3]> {
    use geo::prelude::*;
    use geo::geometry::{LineString, Polygon};
    use geo_rasterize::{LabelBuilder, MergeAlgorithm};

    let minx = points.iter().fold(f32::INFINITY, |a, &b| a.min(b.0)).floor() as usize;
    let miny = points.iter().fold(f32::INFINITY, |a, &b| a.min(b.1)).floor() as usize;
    let maxx = points.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.0)).ceil() as usize;
    let maxy = points.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.1)).ceil() as usize;

    let poly = &Polygon::new(LineString::from(points.into_iter().map(|p| (p.0 - minx as f32, p.1 - miny as f32)).collect::<Vec<_>>()), vec![]);

    (0..maxx-minx).into_par_iter()
        .map(move |xi| {
            (0..maxy-miny).into_par_iter().map(move |yi| {
                let (x, y) = (xi as f32, yi as f32);
                let px = Polygon::new(LineString::from(vec![(x, y), (x + 1.0, y), (x + 1.0, y + 1.0), (x, y + 1.0)]), vec![]);
                let weight = px.intersection(poly).signed_area();
                if weight > 0.0 {
                    image.get_pixel_checked((xi + minx) as u32, (yi + miny) as u32).map(|v| (weight, v.0.map(|c| c * weight)))
                } else {
                    None
                }
            })
            .flatten()
        })
        .flatten()
        .reduce_with(|a, b| (a.0 + b.0, [a.1[0] + b.1[0], a.1[1] + b.1[1], a.1[2] + b.1[2]]))
        .map(|(total, v)| v.map(|x| x / total))
}

fn get_colors_avg(image: &image::Rgb32FImage, rays: u32, gates: u32, x: f32, y: f32, size: f32) -> HashMap<usize, [f32; 3]> {
    let mut colors = HashMap::new();

    for ray in 0..rays {
        let a1 = 2.0 * std::f32::consts::PI * (ray as f32 - 0.5) / rays as f32;
        let a2 = 2.0 * std::f32::consts::PI * (ray as f32 + 0.5) / rays as f32;

        let (sa1, ca1) = a1.sin_cos();
        let (sa2, ca2) = a2.sin_cos();

        for gate in 0..gates {
            let r1 = size * gate as f32 / gates as f32;
            let r2 = size * (gate + 1) as f32 / gates as f32;

            let pts = [(r1 * ca1, r1 * sa1), (r2 * ca1, r2 * sa1), (r2 * ca2, r2 * sa2), (r1 * ca2, r1 * sa2)].map(|p| (p.0 + x * image.width() as f32, p.1 + y * image.height() as f32));

            if pts.iter().all(|p| 0.0 > p.0 || p.0 > image.width() as f32) || pts.iter().all(|p| 0.0 > p.1 || p.1 > image.height() as f32) {
                continue;
            }

            let idx = gate as usize + ray as usize * gates as usize;

            if let Some(color) = weight(image, pts.into()) {
                colors.insert(idx, color);
            }
        }
    }

    colors
}

fn get_colors_nearest(image: &image::Rgb32FImage, rays: u32, gates: u32, x: f32, y: f32, size: f32) -> HashMap<usize, [f32; 3]> {
    let mut colors = HashMap::new();

    for ray in 0..rays {
        let azimuth = 360.0 * ray as f32 / rays as f32;

        for gate in 0..gates {
            let x = (azimuth.to_radians().cos() * gate as f32 * (size / gates as f32) + x * image.width() as f32).round() as u32;
            let y = (azimuth.to_radians().sin() * gate as f32 * (size / gates as f32) + y * image.height() as f32).round() as u32;

            if 0 < x && x < image.width() && 0 < y && y < image.height() {
                let idx = gate as usize + ray as usize * gates as usize;
                colors.insert(idx, image[(x, y)].0);
            }
        }
    }

    colors
}

fn write_image(image: impl AsRef<std::path::Path>, (x, y): (f32, f32), (rays, gates): (u32, u32), radius: f32, fit: Fit, kmeans: bool, antialias: bool) -> Result<(), Box<dyn std::error::Error>> {
    let image = image::open(image)?.to_rgb32f();
    
    let size = match fit {
        Fit::Max => (image.width() as f32).hypot(image.height() as f32) / 2.0,
        Fit::Min => std::cmp::min(image.width(), image.height()) as f32 / 2.0,
        Fit::Val(x) => x
    };

    let colors = if antialias {
        get_colors_avg(&image, rays, gates, x, y, size)
    } else {
        get_colors_nearest(&image, rays, gates, x, y, size)
    };

    let data: Vec<_> = colors.values().cloned().flatten().collect();

    let inds = if kmeans {
        write_kmeans_colormap(data)
    } else {
        write_simple_colormap(data)
    };

    let colors = colors.keys().zip(inds).collect::<HashMap<_,_>>();

    let param = silv::ParamDescription {
        meters_to_first_cell: 0.0,
        meters_between_cells: radius / gates as f32 * 1000.0,
        ..Default::default()
    };

    let mut radar = RadarFile {
        name: "KSRX".into(),
        scan_mode: silv::ScanMode::PPI,
        sweeps: Vec::new(),
        params: HashMap::from([("REF".into(), param.clone())]),
    };

    let mut sweep = Sweep {
        elevation: 0.0,
        ..Default::default()
    };

    for rayi in 0..rays {
        let azimuth = 360.0 * rayi as f32 / rays as f32;

        let mut ray = Ray {
            azimuth: azimuth + 90.0,
            data: HashMap::from([("REF".into(), Vec::new())]),
            ..Default::default()
        };

        for gate in 0..gates {
            let color_idx = if let Some(o) = colors.get(&((gate + rayi * gates) as usize)) {
                *o as f64 / 2.0 - 32.0
            } else {
                -999.0
            };

            ray.data.get_mut("REF").unwrap().push(color_idx);
        }

        // Padding gates because GR2 clips some?????
        for _ in 0..45 {
            ray.data.get_mut("REF").unwrap().push(-999.0);
        }
        
        sweep.rays.push(ray);
    }

    radar.sweeps.push(sweep);
    silv::write(radar, ".", &silv::RadyOptions::default());

    Ok(())
}

const ANNEALING_PAR_N: usize = 30;
const ANNEALING_RUNS: usize = 500_000;
const KMEANS_RUNS: usize = 30;
const K: usize = 20; // Max: 254

fn parse_args() {
    use clap::{Command, Arg};

    let m = Command::new("gr2-image")
        .author("SuperWinner50")
        .version("1.0.0")
        .about("A program to convert images into a radar-readable format (Specifically for GR2).")
        .args([
            Arg::new("position")
                .short('p')
                .long("pos")
                .help("Image coordinates of the position of the radar (0-1)")
                .num_args(2)
                .value_names(["x", "y"])
                .default_values(["0.5", "0.5"]),
            Arg::new("resolution")
                .short('r')
                .long("res")
                .help("Number of rays and gates")
                .num_args(2)
                .value_names(["rays", "gates"])
                .default_values(["1000", "1000"]),
            Arg::new("size")
                .short('s')
                .long("size")
                .help("Scale of radar from the image, with options to fit entire image or largest possible without borders. Options: max, min, or size in pixels")
                .default_value("max"),
            Arg::new("radius")
                .short('R')
                .long("radius")
                .help("Radius of image in km")
                .default_value("100"),
            Arg::new("kmeans")
                .short('k')
                .long("kmeans")
                .action(clap::ArgAction::SetTrue)
                .help("Use a kmeans algorithm to make a custom colormap based off of the input image. Multithreaded."),
            Arg::new("antialias")
                .short('a')
                .long("aa")
                .action(clap::ArgAction::SetTrue)
                .help("Average colors inside of each cell. May add a significant amount of processing time. Multithreaded."),
            Arg::new("files")
                .required(true)
        ]).get_matches();

    let pos = m.get_many::<String>("position").unwrap().map(|v| v.parse::<f32>().unwrap()).collect::<Vec<_>>();
    let res = m.get_many::<String>("resolution").unwrap().map(|v| v.parse::<u32>().unwrap()).collect::<Vec<_>>();
    
    let fit = match m.get_one::<String>("size").unwrap().as_str() {
        "max" => Fit::Max,
        "min" => Fit::Min,
        v => Fit::Val(v.parse().unwrap())
    };

    let radius = m.get_one::<String>("radius").unwrap().parse().unwrap();
    let im = m.get_one::<String>("files").unwrap();

    match write_image(&im, (pos[0], pos[1]), (res[0], res[1]), radius, fit, m.get_flag("kmeans"), m.get_flag("antialias")) {
        Err(e) => println!("Error {e} occured while reading {im:?}"),
        _ => ()
    }
    
}

fn main() {
    parse_args();
}
