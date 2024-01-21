mod algo;
mod in_data;
mod knapsack;
mod pretty;
mod solve;

use crate::algo::{compute_schedule, InputData, Instance};
use crate::in_data::parse;
use crate::pretty::pretty;

fn main() {
    let in_data = parse();

    let InputData {
        epsilon,
        instance:
            Instance {
                ref jobs,
                machine_count,
                resource_limit,
            },
    } = in_data;
    println!("Scheduling {} jobs on {} machines with a resource limit of {} with epsilon={} close to OPT", jobs.len(), machine_count, resource_limit, epsilon);

    let schedule = compute_schedule(in_data);

    println!("Done, result is:");
    println!("{}", pretty(schedule));
}
