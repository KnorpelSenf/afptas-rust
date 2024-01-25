mod algo;
mod in_data;
mod pretty;

use crate::algo::{compute_schedule, Instance};
use crate::in_data::parse;
use crate::pretty::pretty;

fn main() {
    let instance = parse();

    let Instance {
        epsilon,
        machine_count,
        resource_limit,
        ref jobs,
    } = instance;
    println!("Scheduling {} jobs on {} machines with a resource limit of {} with epsilon={} close to OPT", jobs.len(), machine_count, resource_limit, epsilon);

    let schedule = compute_schedule(instance);

    println!("Done, result is:\n{}", pretty(schedule));
}
