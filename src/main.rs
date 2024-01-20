mod algo;
mod ilp;
mod in_data;
mod knapsack;
mod max_min;
mod pretty;

use crate::algo::compute_schedule;
use crate::in_data::parse;
use crate::pretty::pretty;

fn main() {
    let in_data = parse();
    println!("{:#?}", in_data);

    println!("Computing schedule");
    let schedule = compute_schedule(in_data);

    println!("Done, result is:");
    println!("{}", pretty(schedule));
}
