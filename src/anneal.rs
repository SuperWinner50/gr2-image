use palette::{Lab, Srgb, color_difference::Ciede2000, FromColor};
use std::io::Write;
use rand::Rng;
use rayon::prelude::*;

const ANNEALING_PAR_N: usize = 30;
const ANNEALING_RUNS: usize = 500_000;

fn total_cost(colors: &Vec<(usize, Lab)>) -> f32 {
    colors.windows(2).map(|c| c[0].1.difference(c[1].1)).sum::<f32>()
}

fn cost_diff(colors: &Vec<(usize, Lab)>, a: usize, b: usize) -> f32 {
    let mut end = 0.0;
    if a > 0 {
        end += colors[a - 1].1.difference(colors[b].1) - colors[a - 1].1.difference(colors[a].1);
    }

    // These cancel out if a and b are adjacent
    if a.abs_diff(b) != 1 {
        if a < colors.len() - 1 {
            end += colors[b].1.difference(colors[a + 1].1) - colors[a].1.difference(colors[a + 1].1);
        }
        
        if b > 0 {
            end += colors[b - 1].1.difference(colors[a].1) - colors[b - 1].1.difference(colors[b].1);
        }
    }

    if b < colors.len() - 1 {
        end += colors[a].1.difference(colors[b + 1].1) - colors[b].1.difference(colors[b + 1].1);
    }

    end
}

fn anneal(colors: &Vec<(usize, Lab)>) -> (Vec<(usize, Lab)>, f32)  {    
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

pub fn write_annealing_colormap(cmap: Vec<Lab>, inds: Vec<u8>) -> Vec<u8> {
    let res = cmap.into_iter().enumerate().collect::<Vec<_>>();

    let (sorted, _) = (0..ANNEALING_PAR_N).into_par_iter()
        .map(|_| anneal(&res))
        .reduce_with(|a, b| if a.1 < b.1 { a } else { b })
        .unwrap();

    let idxs = sorted.iter().map(|(i, _)| *i).collect::<Vec<_>>();

    let rgb = sorted
        .into_iter()
        .map(|(_, x)| Srgb::from_color(x).into_format())
        .collect::<Vec<Srgb<u8>>>();

    let mut writer = std::fs::File::create("colormap.pal").unwrap();

    writeln!(writer, "Product: BR\nUnits:  DBZ\nStep:   10\n").unwrap();

    for i in 0..rgb.len() {
        let (r, g, b) = rgb[i].into_components();
        let idx = crate::offset(i as f32);

        writeln!(writer, "Color: {idx} {r} {g} {b} {r} {g} {b}").unwrap();
    }

    let mut idxs2 = (0..rgb.len()).collect::<Vec<_>>();
    idxs2.sort_by_key(|i| idxs[*i]);

    inds.iter().cloned().map(|v| idxs2[v as usize] as u8).collect()
}