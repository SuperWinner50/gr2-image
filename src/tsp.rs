use palette::color_difference::Ciede2000;
use palette::{Lab, Srgb, FromColor};

use std::io::{Write, BufRead, BufReader};

use lkh_rs::run;

static PARAMS: &str = "\
TOUR_FILE = problem.out
RUNS = 2
MOVE_TYPE = 5
";

fn get_problem(nodes: &Vec<Lab>) -> String {
    let n = nodes.len();
    
    let mut problem = format!("\
NAME : color-smoother
TYPE: TSP
DIMENSION: {}
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: LOWER_ROW
EDGE_WEIGHT_SECTION:\n", n + 1);

    // Dummy node at end with 0 distance
    for i in 0..=n {
        for j in 0..i {
            let d = if i == n {
                0.0
            } else {
                nodes[i].difference(nodes[j])
            };

            problem += &format!("{:.0} ", d * 100.0);
        }
    }

    problem
}

fn parse_output() -> std::io::Result<Vec<usize>> {
    let mut inds = Vec::new();
    let file = BufReader::new(std::fs::File::open("problem.out")?);

    let mut read = false;

    for line in file.lines() {
        let line = line?;

        match &line {
            x if x.starts_with("TOUR_SECTION") => { read = true; continue },
            x if x.starts_with("EOF") || x.starts_with("-1") => break,
            _ => ()
        }

        if read {
            let id = line.parse::<usize>().unwrap() - 1;
            inds.push(id);
        }
    }

    Ok(inds)
}

fn solve(nodes: &Vec<Lab>) -> Vec<(usize, Lab)> {
    let problem = get_problem(nodes);
    run(PARAMS, &problem);

    let mut out = parse_output().unwrap();
    std::fs::remove_file("problem.out").unwrap();

    let p = out.iter().position(|i| *i == nodes.len()).unwrap();
    out.rotate_left(p);
    out.into_iter().skip(1).map(|i| (i, nodes[i])).collect()
}

pub fn write_tsp_colormap(cmap: Vec<Lab>, inds: Vec<u8>) -> Vec<u8> {
    let sorted = solve(&cmap);

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
