use crate::algo::Job;
use good_lp::{
    constraint, default_solver, variable, variables, Expression, ProblemVariables, Solution,
    SolverModel, Variable,
};

pub fn max_min(
    epsilon_prime: f64,
    jobs: &Vec<Job>,
    narrow_threshold: f64,
    wide_job_max_count: i32,
    machine_count: i32,
    resource_limit: f64,
) -> Vec<f64> {
    let rho = epsilon_prime / (1.0 + epsilon_prime);
    let n = jobs.len();
    println!("Solving max-min for {} jobs with rho={}", n, rho);
    let t = rho / 6.0;
    // compute initial solution;
    println!("Computing initial solution");
    let mut solution = initial(
        jobs,
        narrow_threshold,
        wide_job_max_count,
        machine_count,
        resource_limit,
    ); // \v{x}
    println!("Initial solution is {:?}", solution);

    // iterate
    loop {
        // price vector
        let price = compute_price(&solution);
        println!("++ Starting iteration with price {:?}", price);
        // solve block problem
        let max = solve_block_problem_ilp(
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

macro_rules! unit {
    ( $i:expr, $m:expr ) => {{
        let mut temp_vec = vec![0.0f64; $m];
        temp_vec[$i] = 1.0f64;
        temp_vec
    }};
}

fn initial(
    jobs: &Vec<Job>,
    narrow_threshold: f64,
    wide_job_max_count: i32,
    machine_count: i32,
    resource_limit: f64,
) -> Vec<f64> {
    let m = jobs.len();
    let scale = 1.0f64 / (m as f64);
    let units = (0..m).map(|i| unit!(i, m));
    units
        .map(|e| {
            solve_block_problem_ilp(e, jobs, narrow_threshold, 2, machine_count, resource_limit)
                .iter()
                .map(|x| x * scale)
                .collect()
        })
        .fold(vec![0.0; m], |acc: Vec<f64>, x: Vec<f64>| {
            // vec add
            acc.iter().zip(x).map(|(x, y)| x + y).collect::<Vec<f64>>()
        })
}

fn compute_step_length() -> f64 {
    0.0
}

fn compute_price(q: &[f64]) -> Vec<f64> {
    q.to_vec()
}

fn solve_block_problem_ilp(
    q: Vec<f64>,
    jobs: &Vec<Job>,
    narrow_threshold: f64,
    wide_job_max_count: i32,
    machine_count: i32,
    resource_limit: f64,
) -> Vec<f64> {
    let jobs = jobs;
    let mut prob = Ilp::new(machine_count, resource_limit);

    let variables: Vec<_> = q
        .iter()
        .zip(jobs)
        .map(|q_job| {
            let c = ConfigurationCandidate {
                q: *q_job.0,
                p: q_job.1.processing_time,
                r: q_job.1.resource_amount,
                max_a: if q_job.1.is_wide(narrow_threshold) {
                    wide_job_max_count
                } else {
                    1
                },
            };
            println!("Adding variable {:?}", c);
            prob.add(c)
        })
        .collect();

    let solution = prob.find_configuration();
    let a_star: Vec<f64> = variables.iter().map(|&v| solution.value(v)).collect();
    println!(
        "Solved block problem ILP (n={}) for {:?} with {:?}",
        jobs.len(),
        q,
        a_star
    );
    a_star
}

#[derive(Debug)]
struct ConfigurationCandidate {
    q: f64,
    p: f64,
    r: f64,
    max_a: i32,
}

struct Ilp {
    vars: ProblemVariables,
    machine_count: Expression,
    resource_amount: Expression,
    machine_limit: i32,
    resource_limit: f64,
    objective: Expression,
}

impl Ilp {
    fn new(machine_limit: i32, resource_limit: f64) -> Ilp {
        Ilp {
            vars: variables!(),
            machine_limit,
            resource_limit,
            machine_count: 0.into(),
            resource_amount: 0.into(),
            objective: 0.into(),
        }
    }

    fn add(&mut self, job: ConfigurationCandidate) -> Variable {
        let ConfigurationCandidate { q, p, r, max_a } = job;
        let a = self.vars.add(variable().integer().min(0).max(max_a));
        self.objective += (q / p) * a;
        self.machine_count += a;
        self.resource_amount += r * a;
        a
    }

    fn find_configuration(self) -> impl Solution {
        let mut model = self
            .vars
            .maximise(self.objective)
            .using(default_solver)
            .with(constraint!(self.machine_count <= self.machine_limit))
            .with(constraint!(self.resource_amount <= self.resource_limit));
        model.set_parameter("log", "0"); // suppress log output by solver
        model.solve().expect("no ILP solution")
    }
}
