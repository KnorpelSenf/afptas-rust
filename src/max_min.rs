use crate::{algo::Job, ilp::solve_ilp};

pub fn max_min(
    rho: f64,
    jobs: &Vec<Job>,
    narrow_threshold: f64,
    wide_job_max_count: i32,
    machine_count: i32,
    resource_limit: f64,
) -> Vec<f64> {
    let n = jobs.len();
    println!("Solving max-min for {} jobs with rho={}", n, rho);
    // compute initial solution;
    let mut solution = initial(n); // \v{x}
    println!("Initial solution is {:?}", solution);

    // iterate
    loop {
        // price vector
        let price = compute_price(&solution);
        println!("++ Starting iteration with price {:?}", price);
        // solve block problem
        let max = solve_block_problem(
            price,
            jobs,
            narrow_threshold,
            wide_job_max_count,
            machine_count,
            resource_limit,
        );
        println!("Received block problem solution {:?}", max);
        // update solution = ((1-tau) * solution) + (tau * solution)
        let tau = compute_step_length();
        let one_minus_tau = 1.0 - tau;
        for i in 0..n {
            solution[i] = one_minus_tau * solution[i] + tau * solution[i]
        }
        println!(
            "Updated solution with step length tau={} to be {:?}",
            tau, solution
        );
        break;
    }
    println!("Max-min solved with {:?}", solution);
    solution
}

fn initial(n: usize) -> Vec<f64> {
    vec![1.0f64; n]
}

fn compute_step_length() -> f64 {
    0.0
}

fn compute_price(q: &[f64]) -> Vec<f64> {
    q.to_vec()
}

fn solve_block_problem(
    s: Vec<f64>,
    jobs: &Vec<Job>,
    narrow_threshold: f64,
    wide_job_max_count: i32,
    machine_count: i32,
    resource_limit: f64,
) -> Vec<f64> {
    println!("Solving block problem for {} jobs", jobs.len());
    let solution = solve_ilp(
        s.clone(),
        jobs,
        narrow_threshold,
        wide_job_max_count,
        machine_count,
        resource_limit,
    );
    println!("Solved block problem with {:?}", solution);
    solution
}
