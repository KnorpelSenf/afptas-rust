use std::env::args;
use std::io::stdin;
use std::io::BufRead;

use crate::in_data::InputData;
use crate::in_data::Instance;
use crate::in_data::Job;
mod in_data;

fn main() {
    // get the first 3 command line arguments, which are numbers and than read all input lines with read_line inside a for loop and save the result in an array
    let (m, r, e) = (
        args().nth(1).unwrap().parse::<u32>().unwrap(),
        args().nth(2).unwrap().parse::<f64>().unwrap(),
        args().nth(3).unwrap().parse::<f64>().unwrap(),
    );
    let mut jobs = Vec::new();
    for line in stdin().lock().lines() {
        let line = line.unwrap();
        let mut line = line.split(",");
        let (x, y) = (
            line.next().unwrap().parse::<f64>().unwrap(),
            line.next().unwrap().parse::<f64>().unwrap(),
        );
        jobs.push((x, y));
    }

    let in_data = InputData {
        epsilon: e,
        instance: Instance {
            machine_count: m,
            resource_limit: r,
            jobs: Box::from(
                jobs.iter()
                    .enumerate()
                    .map(|(id, (p, r))| Job {
                        id: id as u32,
                        processing_time: *p,
                        resource_amount: *r,
                    })
                    .collect::<Vec<Job>>(),
            ),
        },
    };

    println!("m:{}, R:{}, e:{}", m, r, e);
    println!("{:?}", in_data);
}
