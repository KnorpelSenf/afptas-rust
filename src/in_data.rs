use std::fs::File;
use std::io;
use std::io::stdin;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Lines;
use std::iter::Iterator;
use std::path::Path;

use clap::Parser;

use crate::algo::InputData;
use crate::algo::Instance;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    epsilon: f64,

    #[arg(short, long)]
    machines: u32,

    #[arg(short, long)]
    resource_limit: f64,

    #[arg(short, long)]
    job_file: String,
}

pub fn parse() -> InputData {
    let args = Args::parse();

    let job_file: Vec<String> = if args.job_file == "-" {
        stdin().lines().filter_map(|line| line.ok()).collect::<Vec<String>>()
    } else {
        BufReader::new(File::open(args.job_file).expect("Could not open job file"))
            .lines()
            .filter_map(|line| line.ok())
            .collect::<Vec<String>>()
    };
    let jobs = job_file.enumerate().map(|(index, line)| {
        let line_number: u32 = index + 1;
        let cols = line.split(",").take(2).enumerate().collect();
        let p = cols.get(0);
        let r = cols.get(1);
        (index, p, r)
    });

    InputData {
        epsilon: args.epsilon,
        instance: Instance {
            machine_count: args.machines,
            resource_limit: args.resource_limit,
            jobs: Box::from([]),
        },
    }
    // let arg_list = args();
    // let machine_count = arg_list.nth(1);

    // // get the first 3 command line arguments, which are numbers and than read all input lines with read_line inside a for loop and save the result in an array
    // let (m, r, e) = (
    //     args().nth(1).unwrap().parse::<u32>().unwrap(),
    //     args().nth(2).unwrap().parse::<f64>().unwrap(),
    //     args().nth(3).unwrap().parse::<f64>().unwrap(),
    // );
    // let mut jobs = Vec::new();
    // for line in stdin().lock().lines() {
    //     let line = line.unwrap();
    //     let mut line = line.split(",");
    //     let (x, y) = (
    //         line.next().unwrap().parse::<f64>().unwrap(),
    //         line.next().unwrap().parse::<f64>().unwrap(),
    //     );
    //     jobs.push((x, y));
    // }

    // InputData {
    //     epsilon: e,
    //     instance: Instance {
    //         machine_count: m,
    //         resource_limit: r,
    //         jobs: Box::from(
    //             jobs.iter()
    //                 .enumerate()
    //                 .map(|(id, (p, r))| Job {
    //                     id: id as u32,
    //                     processing_time: *p,
    //                     resource_amount: *r,
    //                 })
    //                 .collect::<Vec<Job>>(),
    //         ),
    //     },
    // }
}

fn read_job_file<P>(filename: P) -> Lines<BufReader<File>>
where
    P: AsRef<Path>,
{
}
