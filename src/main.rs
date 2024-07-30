mod algo;
mod in_data;
mod pretty;

use std::time::Instant;

use crate::algo::{compute_schedule, Instance};
use crate::in_data::parse;
use crate::pretty::{display, pretty};

fn main() {
    let instance = parse();

    let Instance {
        epsilon,
        machine_count,
        resource_limit,
        ref jobs,
    } = instance;
    let job_count = jobs.len();
    println!("Scheduling {job_count} jobs on {machine_count} machines with a resource limit of {resource_limit} with epsilon={epsilon} close to OPT");

    let start = Instant::now();
    let schedule = compute_schedule(instance);
    let duration = start.elapsed();

    println!("Done in {:?}.", duration);
    if job_count <= 1000 {
        println!("Result is (prettified):");
        println!("{}", pretty(schedule));
    } else {
        println!("Result is:");
        println!("{}", display(schedule));
    }
    println!("{machine_count} machines.");
}
