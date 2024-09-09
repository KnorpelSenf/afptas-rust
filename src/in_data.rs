use std::fs::File;
use std::io::stdin;
use std::io::BufRead;
use std::io::BufReader;

use std::iter::Iterator;

use clap::Parser;

use crate::algo::Instance;
use crate::algo::InstanceJob;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    epsilon: f64,

    #[arg(short, long)]
    machines: i32,

    #[arg(short, long)]
    resource_limit: f64,

    #[arg(short, long)]
    job_file: String,
}

pub fn parse() -> Instance {
    let args = Args::parse();
    println!("Parsing input data");

    let is_stdin = args.job_file == "-";
    let job_file: Vec<String> = if is_stdin {
        stdin()
            .lines()
            .filter_map(|line| line.ok())
            .collect::<Vec<String>>()
    } else {
        BufReader::new(File::open(args.job_file.clone()).expect("Could not open job file"))
            .lines()
            .filter_map(|line| line.ok())
            .collect::<Vec<String>>()
    };
    let jobs = job_file
        .iter()
        .skip(1) // drop column headers
        .enumerate()
        .map(|(index, line)| {
            let mut cols = line.split(",").take(2);
            let p = cols
                .next()
                .expect(format!("missing col 0 in row {}", index).as_str())
                .parse::<f64>()
                .expect(format!("cannot parse col 0 as int in row {}", index).as_str());
            let r = cols
                .next()
                .expect(format!("missing col 1 in row {}", index).as_str())
                .parse::<f64>()
                .expect(format!("cannot parse col 1 as int in row {}", index).as_str());
            if r > args.resource_limit {
                let line = index + 1; // column header offset
                let pos = if is_stdin {
                    format!("on input line {}", line)
                } else {
                    format!("at {}:{}", args.job_file, line)
                };
                panic!(
                    "No possible solution, job {} needs {} resources but the resource limit is {}",
                    pos, r, args.resource_limit
                );
            }
            InstanceJob {
                processing_time: p,
                resource_amount: r,
            }
        })
        .collect::<Vec<_>>();

    Instance {
        epsilon: args.epsilon,
        machine_count: args.machines,
        resource_limit: args.resource_limit,
        jobs: Box::from(jobs),
    }
}
