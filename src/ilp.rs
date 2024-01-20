use std::error::Error;

use good_lp::{variable, variables, default_solver};

use crate::algo::Job;

pub fn example() -> Result<(), Box<dyn Error>> {
    variables! {
        vars:
          a <= 4;
          0 <= b (integer) <= 1;
    } // variables can also be added dynamically
    let solution = vars
        .maximise(a + b)
        .using(default_solver) // multiple solvers available
        .solve()?;
    println!("a={}   b={}", solution.value(a), solution.value(b));
    println!("a + b = {}", solution.eval(a + b));
    Ok(())
}

pub fn other_example() -> Result<(), Box<dyn Error>> {
    // Create variables in a readable format with a macro...
    variables! {
       vars:
           a <= 1;
           2 <= b <= 4;
    }
    // ... or add variables programmatically
    vars.add(variable().min(2).max(9));

    let solution = vars
        .maximise(10 * (a - b / 5) - b)
        .using(default_solver)
        .with(a + 2. << b) // or (a + 2).leq(b)
        .with(1 + a >> 4. - b)
        // .with(constraint!(1 + a >= 4. - b))
        .solve()?;

    assert_eq!(solution.value(a), 1.);
    assert_eq!(solution.value(b), 3.);
    Ok(())
}

pub fn solve_ilp(
    q: Vec<f64>,
    jobs: &Vec<Job>,
    narrow_threshold: f64,
    wide_job_max_count: u32, // should be 1 / epsilon_prime
    machine_count: u32,
    resource_limit: f64,
) {
    // let replicated_wide_jobs = wide_jobs
    //     .into_iter()
    //     .flat_map(|job| vec![job; wide_job_max_count])
    //     .collect::<Vec<_>>();
    // let initial_to_replicated = |i: usize| i * wide_job_max_count;
    // let replicated_to_initial = |i: usize| i / wide_job_max_count;

    let mut problem = variables!();
    let vars = jobs
        .iter()
        .map(|job| {
            variable()
                .integer()
                .min(0)
                .max(if job.is_wide(narrow_threshold) {
                    wide_job_max_count
                } else {
                    1
                })
        })
        .collect::Vec<_>();
    for var in vars {
        problem.add(var);
    }
    // problem.maximise();
}
