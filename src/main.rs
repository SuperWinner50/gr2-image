use std::collections::HashMap;
use std::io::Write;

use silv::{RadarFile, Sweep, Ray};
use rayon::prelude::*;

use palette::{Lab, Srgb, IntoColor, FromColor};

mod kmeans;
mod anneal;
mod tsp;
use kmeans::*;

fn offset(x: f32) -> f32 {
    x / 2.0 - 32.0
}

fn get_simple_colormap(data: Vec<f32>) -> (Vec<Lab>, Vec<u8>) {
    let mut colors = Vec::new();
    let mut indices = Vec::new();

    let rx = 6;
    let gx = 6;
    let bx = 6;

    for c3 in 0..bx {
        for c2 in 0..gx {
            for c1 in 0..rx {
                let r = c1 as f32 / (rx - 1) as f32;
                let g = c2 as f32 / (gx - 1) as f32;
                let b = c3 as f32 / (bx - 1) as f32;
                
                let color = Srgb::new(r, g, b).into_color();
                colors.push(color);
            }
        }
    }

    for color in data.chunks(3) {
        let r = rx as f32 * clamp(color[0]);
        let g = gx as f32 * clamp(color[1]);
        let b = bx as f32 * clamp(color[2]); 

        indices.push(r as u8 + g as u8 * rx + b as u8 * rx * bx);
    }

    (colors, indices)
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
    use geo_clipper::Clipper;
    use geo::{Area, geometry::{LineString, Polygon}};

    let minx = points.iter().fold(f32::INFINITY, |a, &b| a.min(b.0)).floor() as usize;
    let miny = points.iter().fold(f32::INFINITY, |a, &b| a.min(b.1)).floor() as usize;
    let maxx = points.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.0)).ceil() as usize;
    let maxy = points.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.1)).ceil() as usize;

    let pts = points.iter().map(|p| (p.0 - minx as f32, p.1 - miny as f32)).collect::<Vec<_>>();
    let poly = &Polygon::new(LineString::from(pts.clone()), vec![]);

    (0..maxx-minx).into_par_iter()
        .flat_map(move |xi| {
            (0..maxy-miny).into_par_iter().map(move |yi| {
                let (x, y) = (xi as f32, yi as f32);                
                let px = Polygon::new(LineString::from(vec![(x, y), (x + 1.0, y), (x + 1.0, y + 1.0), (x, y + 1.0)]), vec![]);
        
                let weight = px.intersection(poly, 100000.0).signed_area();
                
                if weight > 0.0 {
                    image.get_pixel_checked((xi + minx) as u32, (yi + miny) as u32).map(|v| (weight, v.0.map(|c| c * weight)))
                } else {
                    None
                }
            })
            .flatten()
        })
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

            if pts.iter().all(|p| 0.0 > p.0 || p.0 >= image.width() as f32) || pts.iter().all(|p| 0.0 > p.1 || p.1 >= image.height() as f32) {
                continue;
            }

            let idx = gate as usize + ray as usize * gates as usize;

            if let Some(color) = weight(image, pts.into()) {
                colors.insert(idx, color);
            } else {
                println!("No weight value value found; Points: {pts:?}");
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

pub fn write_colormap(cmap: Vec<Lab>, inds: Vec<u8>) -> Vec<u8> {
    let rgb = cmap
        .into_iter()
        .map(|x| Srgb::from_color(x).into_format())
        .collect::<Vec<Srgb<u8>>>();

    let mut writer = std::fs::File::create("colormap.pal").unwrap();

    writeln!(writer, "Product: BR\nUnits:  DBZ\nStep:   10\n").unwrap();

    for i in 0..rgb.len() {
        let (r, g, b) = rgb[i].into_components();
        let idx = i as f32 / 2.0 - 32.0;

        writeln!(writer, "Color: {idx} {r} {g} {b} {r} {g} {b}").unwrap();
    }

    inds
}

fn write_image(image: impl AsRef<std::path::Path>, (x, y): (f32, f32), (rays, gates): (u32, u32), radius: f32, fit: Fit, kmeans: Option<usize>, antialias: bool, smoothing: usize) -> Result<(), Box<dyn std::error::Error>> {
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

    let (cmap, inds) = if let Some(k) = kmeans {
        get_kmeans_colormap(data, k)
    } else {
        get_simple_colormap(data)
    };

    let inds = match smoothing {
        0 => write_colormap(cmap, inds),
        1 => anneal::write_annealing_colormap(cmap, inds),
        2 => tsp::write_tsp_colormap(cmap, inds),
        _ => panic!("Bad")
    };

    let colors = colors.keys().zip(inds).collect::<HashMap<_,_>>();

    let param = silv::ParamDescription {
        meters_to_first_cell: 0.0,
        meters_between_cells: radius / gates as f32 * 1000.0,
        ..Default::default()
    };

    let mut radar = RadarFile {
        name: "KCYS".into(),
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
                offset(*o as f32) as f64
            } else {
                -999.0
            };

            ray.data.get_mut("REF").unwrap().push(color_idx);
        }

        // Padding gates because GR2 clips some
        for _ in 0..45 {
            ray.data.get_mut("REF").unwrap().push(-999.0);
        }
        
        sweep.rays.push(ray);
    }

    radar.sweeps.push(sweep);
    silv::write(radar, ".", &silv::RadyOptions::default());

    Ok(())
}

fn parse_args() {
    use clap::{Command, Arg, builder::PossibleValue, ArgAction};

    let m = Command::new("gr2-image")
        .author("SuperWinner50")
        .version("1.0.0")
        .about("A program to convert images into a radar-readable format (Specifically for GR2)")
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
            Arg::new("fit")
                .short('f')
                .long("fit")
                .next_line_help(true)
                .help("How to fit the radar to the image\n\
                       Options: max, min, pixel radius")
                .default_value("max"),
            Arg::new("radius")
                .short('R')
                .long("radius")
                .help("Radius of image in km")
                .default_value("100"),
            Arg::new("kmeans")
                .short('k')
                .long("kmeans")
                .num_args(0..=1)
                .require_equals(true)
                .default_missing_value("40")
                .help("Number of clusters to use for kmeans algorithm. Max value: 254")
                .value_parser(clap::value_parser!(u32).range(1..=254)),
            Arg::new("antialias")
                .short('a')
                .long("aa")
                .action(ArgAction::SetTrue)
                .help("Average colors inside of each cell. May add a significant amount of processing time"),
            Arg::new("smoothing")
                .short('S')
                .long("smooth")
                .help("Smooths out the colormap using using the following algorithms")
                .value_parser([
                    PossibleValue::new("0").help("Do not use a smoothing algorithm"),
                    PossibleValue::new("1").help("Use a simulated annealing algorithm"),
                    PossibleValue::new("2").help("Use a travelling salesman solver")
                ])
                .default_value("0"),
            Arg::new("files")
                .required(true)
        ]).get_matches();

    let pos = m.get_many::<String>("position").unwrap().map(|v| v.parse::<f32>().unwrap()).collect::<Vec<_>>();
    let res = m.get_many::<String>("resolution").unwrap().map(|v| v.parse::<u32>().unwrap()).collect::<Vec<_>>();
    let k = m.get_one::<u32>("kmeans").map(|&v| v as usize);
    let smoothing = m.get_one::<String>("smoothing").unwrap().parse::<usize>().unwrap();
    
    let fit = match m.get_one::<String>("fit").unwrap().as_str() {
        "max" => Fit::Max,
        "min" => Fit::Min,
        v => Fit::Val(v.parse().unwrap())
    };

    let radius = m.get_one::<String>("radius").unwrap().parse().unwrap();
    let im = m.get_one::<String>("files").unwrap();

    match write_image(&im, (pos[0], pos[1]), (res[0], res[1]), radius, fit, k, m.get_flag("antialias"), smoothing) {
        Err(e) => println!("Error {e} occured while reading {im:?}"),
        _ => ()
    }
}

fn main() {
    parse_args();
}
