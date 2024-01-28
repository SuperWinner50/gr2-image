use std::collections::HashMap;
use std::io::Write;

use silv::{RadarFile, Sweep, Ray};
use rand_distr::StandardNormal;
use rand::{thread_rng, Rng};

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

fn write_kmeans_colormap(data: Vec<f32>) -> Vec<u8> {
    use palette::cast::from_component_slice;
    use palette::{FromColor, IntoColor, Lab, Srgb};
    use kmeans_colors::{get_kmeans_hamerly, Calculate, Kmeans, MapColor, Sort};
    use rayon::prelude::*;

    const RUNS: usize = 20;
    const K: usize = 40;

    let lab: Vec<Lab> = from_component_slice::<Srgb<f32>>(&data)
        .iter()
        .map(|x| x.into_format().into_color())
        .collect();

    let mut seed = rand::random::<u32>() as u64;
    let result = (0..RUNS).into_par_iter()
        .map(|i| {
            get_kmeans_hamerly(
                K,
                40,
                0.5,
                true,
                &lab,
                seed + i as u64,
            )
        })
        .reduce_with(|a, b| if a.score < b.score { a } else { b })
        .unwrap();

    println!("Min score: {}", result.score);

    let rgb = &result.centroids
        .iter()
        .map(|x| Srgb::from_color(*x).into_format())
        .collect::<Vec<Srgb<u8>>>();

    let mut writer = std::fs::File::create("colormap.pal").unwrap();

    writeln!(writer, "Product: BR\nUnits:  DBZ\nStep:   10\n").unwrap();

    for i in 0..K {
        let (r, g, b) = rgb[i].into_components();
        let idx = i as f32 / 2.0 - 32.0;

        writeln!(writer, "Color: {idx} {r} {g} {b} {r} {g} {b}").unwrap();
    }

    result.indices
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

fn intersection

fn write_image(image: impl AsRef<std::path::Path>, (x, y): (f32, f32), (rays, gates): (u32, u32), width: f32, fit: Fit, random: bool, kmeans: bool) -> Result<(), Box<dyn std::error::Error>> {
    let image = image::open(image)?.to_rgb32f();
    
    let size = match fit {
        Fit::Max => (image.width() as f32).hypot(image.height() as f32) / 2.0 * 1.05,
        Fit::Min => std::cmp::min(image.width(), image.height()) as f32 / 2.0,
        Fit::Val(x) => x
    };

    let mut colors = std::collections::HashMap::new();

    use geo::geometry::{Rect, Polygon};

    for xi in 0..image.width() {
        for yi in 0..image.height() {
            let (x, y) = (xi as f32 - x * image.width() as f32, yi as f32 - y * image.height() as f32);
            let r = x.hypot(y);
            let a = y.atan2(x).rem_euclid(std::f32::consts::PI * 2.0);

            let ri = (r / (size / gates as f32)) as usize;

            if ri >= gates as usize {
                continue;
            }

            let ai = (a / (std::f32::consts::PI * 2.0 / rays as f32)) as usize;
            let idx = ri + ai * gates as usize;
            colors.entry(idx).or_insert(Vec::new()).push(image[(xi, yi)].0);
        }
    }

    let colors = colors.into_iter().map(|(k, v)| {
        let len = v.len();
        (k, v.into_iter().reduce(|acc, v| [acc[0] + v[0], acc[1] + v[1], acc[2] + v[2]]).unwrap().map(|v| v / len as f32))
    }).collect::<HashMap<_, _>>();

    let data: Vec<_> = colors.values().cloned().flatten().collect();

    let inds = if kmeans {
        write_kmeans_colormap(data)
    } else {
        write_simple_colormap(data)
    };

    let colors = colors.keys().zip(inds).collect::<HashMap<_,_>>();

    let param = silv::ParamDescription {
        meters_to_first_cell: 0.0,
        meters_between_cells: width,
        ..Default::default()
    };

    let mut radar = RadarFile {
        name: "KDLH".into(),
        scan_mode: silv::ScanMode::PPI,
        sweeps: Vec::new(),
        params: HashMap::from([("REF".into(), param.clone())]),
    };

    let mut sweep = Sweep {
        elevation: 0.0,
        ..Default::default()
    };

    let mut i = 0;
    for rayi in 0..rays {
        let azimuth = 360.0 * rayi as f32 / rays as f32;

        let mut ray = Ray {
            azimuth: azimuth + 90.0,
            data: HashMap::from([("REF".into(), Vec::new())]),
            ..Default::default()
        };

        for gate in 0..gates {
            let x = (azimuth.to_radians().cos() * gate as f32 * (size / gates as f32) + x * image.width() as f32).round() as u32;
            let y = (azimuth.to_radians().sin() * gate as f32 * (size / gates as f32) + y * image.height() as f32).round() as u32;

            let color_idx = if let Some(o) = colors.get(&((gate + rayi * gates) as usize)) {
                *o as f64 / 2.0 - 32.0
            } else {
                -999.0
            };

            ray.data.get_mut("REF").unwrap().push(color_idx);
        }
        
        sweep.rays.push(ray);
    }

    radar.sweeps.push(sweep);
    silv::write(radar, ".", &silv::RadyOptions::default());

    Ok(())
}

fn parse_args() {
    use clap::{Command, Arg};

    let m = Command::new("gr2-image")
        .author("SuperWinner50")
        .version("1.0.0")
        .about("A program to convert images into a radar-readable format.")
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
                .default_values(["1200", "1200"]),
            Arg::new("size")
                .short('s')
                .long("size")
                .help("Scale of radar. Options: max, min, or size in pixels")
                .default_value("max"),
            Arg::new("width")
                .short('w')
                .long("width")
                .help("Gate width")
                .default_value("100"),
            Arg::new("random")
                .short('R')
                .long("random")
                .action(clap::ArgAction::SetTrue)
                .help("Add gaussian randomness for non-kmeans runs")
                .conflicts_with("kmeans"),
            Arg::new("kmeans")
                .short('k')
                .long("use_kmeans")
                .action(clap::ArgAction::SetTrue)
                .help("Uses a kmeans algorithm to make a custom colormap based off of the input image"),
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

    let width = m.get_one::<String>("width").unwrap().parse().unwrap();
    let im = m.get_one::<String>("files").unwrap();

    match write_image(&im, (pos[0], pos[1]), (res[0], res[1]), width, fit, m.get_flag("random"), m.get_flag("kmeans")) {
        Err(e) => println!("Error {e} occured while reading {im:?}"),
        _ => ()
    }
    
}

fn main() {
    parse_args();
}
