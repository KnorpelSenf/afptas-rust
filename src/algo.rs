use good_lp::{
    constraint, default_solver, variable, variables, Expression, ProblemVariables, Solution,
    SolverModel, Variable,
};
use std::{cmp::Ordering, collections::HashMap};

#[derive(Debug)]
pub struct InputData {
    pub epsilon: f64,
    pub instance: Instance,
}

#[derive(Debug)]
pub struct Instance {
    pub machine_count: i32,
    pub resource_limit: f64,
    pub jobs: Box<Vec<Job>>,
}

#[derive(Debug, Clone)]
pub struct Job {
    pub id: i32,
    pub processing_time: f64,
    pub resource_amount: f64,
}
impl Job {
    #[inline]
    pub fn is_wide(&self, threshold: f64) -> bool {
        self.resource_amount >= threshold
    }
}
impl PartialEq for Job {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Job {}

#[derive(Debug)]
struct Configuration {
    index: HashMap<i32, usize>, // job.id -> index in vec
    jobs: Box<Vec<(Job, i32)>>,
}
impl Configuration {
    fn get(&self, job: Job) -> Option<i32> {
        Some(self.jobs[*self.index.get(&job.id)?].1)
    }
    fn set(&mut self, job: Job, count: i32) {
        match self.index.get(&job.id) {
            None => {
                let i = self.jobs.len();
                self.index.insert(job.id, i);
                self.jobs.push((job, count));
            }
            Some(i) => {
                self.jobs[*i] = (job, count);
            }
        }
    }
    fn machines(&self) -> i32 {
        self.jobs.iter().map(|pair| pair.1).sum()
    }
    fn processing_time(&self) -> f64 {
        self.jobs
            .iter()
            .map(|pair| pair.1 as f64 * pair.0.processing_time)
            .sum()
    }
    fn resource_amount(&self) -> f64 {
        self.jobs
            .iter()
            .map(|pair| pair.1 as f64 * pair.0.resource_amount)
            .sum()
    }
    fn is_valid(&self, instance: Instance) -> bool {
        self.machines() <= instance.machine_count
            && self.resource_amount() <= instance.resource_limit
    }
}

#[derive(Debug)]
pub struct Schedule {
    pub mapping: Box<Vec<JobPosition>>,
}

#[derive(Debug)]
pub struct JobPosition {
    pub machine: i32,
    pub starting_time: f64,
}

pub fn compute_schedule(in_data: InputData) -> Schedule {
    println!("Computing schedule");
    let InputData {
        epsilon,
        instance:
            Instance {
                jobs,
                machine_count,
                resource_limit,
            },
    } = in_data;
    let jobs = jobs.into_iter().collect::<Vec<Job>>();

    if 1.0 / epsilon >= machine_count.into() {
        todo!("second case");
    }
    let epsilon_prime = epsilon / 5.0;
    let epsilon_prime_squared = epsilon_prime * epsilon_prime;
    let threshold = epsilon_prime * resource_limit;
    let p_max = jobs
        .iter()
        .max_by(compare_processing_time)
        .expect("no jobs found")
        .processing_time;
    let (narrow_jobs, wide_jobs) = {
        let (narrow_jobs, mut wide_jobs) = jobs
            .clone()
            .into_iter()
            .partition::<Vec<_>, _>(|job| job.is_wide(threshold));
        wide_jobs.sort_by(compare_resource_amount);
        (narrow_jobs, wide_jobs)
    };

    println!(
        "Jobs are partitioned as follows (resource threshold={}):",
        threshold
    );
    println!("Wide {:?}", wide_jobs);
    println!("Narrow {:?}", narrow_jobs);
    let p_w: f64 = wide_jobs.iter().map(|job| job.processing_time).sum();
    let i_sup = create_i_sup(epsilon_prime_squared, p_w, wide_jobs);

    let _ = max_min(
        epsilon,
        &jobs,
        threshold,
        (1.0 / epsilon_prime) as i32,
        machine_count,
        resource_limit,
    );

    Schedule {
        mapping: Box::from(vec![]),
    }
}

fn create_i_sup(epsilon_prime_squared: f64, p_w: f64, wide_jobs: Vec<Job>) -> Vec<Job> {
    let step = epsilon_prime_squared * p_w;
    let mut job_ids = wide_jobs.last().expect("last job").id + 1..;
    let groups = linear_grouping(step, &wide_jobs);
    let additional_jobs = groups
        .into_iter()
        .map(|group| {
            let resource_amount = group
                .into_iter()
                .max_by(compare_resource_amount)
                .expect("empty group")
                .resource_amount;
            Job {
                id: job_ids.next().unwrap(),
                processing_time: step,
                resource_amount,
            }
        })
        .collect::<Vec<_>>();
    println!(
        "Creating {} additional jobs to generate I_sup",
        additional_jobs.len()
    );
    [wide_jobs, additional_jobs].concat()
}

fn linear_grouping(step: f64, jobs: &Vec<Job>) -> Vec<Vec<Job>> {
    // FIXME: Add special handling for the last group since we already know that
    // all remaining jobs will be put into it. Due to floating point
    // imprecision, it might happen that we accidentally open one group to many,
    // containing a single job only, having the size of the floating point
    // rounding error.
    if jobs.len() == 0 {
        return vec![];
    }
    let mut job_ids = 0..;

    let mut groups: Vec<Vec<Job>> = vec![];
    let mut current_group: Vec<Job> = vec![];
    let mut current_processing_time = 0.0f64;
    for job in jobs.iter() {
        let mut remaining_processing_time = job.processing_time;
        loop {
            let remaining_space = (groups.len() + 1) as f64 * step - current_processing_time;
            // Handle last iteration if the job fits entirely
            if remaining_processing_time <= remaining_space {
                let new_job = Job {
                    id: job_ids.next().unwrap(),
                    processing_time: remaining_processing_time,
                    resource_amount: job.resource_amount,
                };
                current_processing_time += remaining_processing_time;
                current_group.push(new_job);
                break;
            }

            // Split off a bit of the job for the current group
            let new_job = Job {
                id: job_ids.next().unwrap(),
                processing_time: remaining_space,
                resource_amount: job.resource_amount,
            };
            current_group.push(new_job);
            groups.push(current_group);

            current_group = vec![];

            current_processing_time += remaining_space;
            remaining_processing_time -= remaining_space;
        }
    }
    groups
}

fn compare_processing_time(job0: &&Job, job1: &&Job) -> Ordering {
    job0.processing_time
        .partial_cmp(&job1.processing_time)
        .expect("invalid processing time")
}

fn compare_resource_amount(job0: &Job, job1: &Job) -> Ordering {
    job0.resource_amount
        .partial_cmp(&job1.resource_amount)
        .expect("invalid resource amount")
}

fn max_min(
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
            // println!("Adding variable {:?}", c);
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
