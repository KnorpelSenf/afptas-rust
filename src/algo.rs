use good_lp::{
    constraint, default_solver, variable, variables, Expression, ProblemVariables, Solution,
    SolverModel, Variable,
};
use std::{
    cmp::Ordering,
    collections::HashMap,
    hash::{Hash, Hasher},
    rc::Rc,
};

// RAW INPUT DATA
#[derive(Debug)]
pub struct Instance {
    pub epsilon: f64,
    pub machine_count: i32,
    pub resource_limit: f64,
    pub jobs: Box<Vec<InstanceJob>>,
}
#[derive(Debug)]
pub struct InstanceJob {
    pub processing_time: f64,
    pub resource_amount: f64,
}

// WORKING DATA
#[derive(Debug, Clone)]
pub struct ProblemData {
    pub epsilon: f64,
    pub epsilon_squared: f64,
    pub epsilon_prime: f64,
    pub epsilon_prime_squared: f64,
    pub one_over_epsilon_prime: i32,
    pub machine_count: i32,
    pub resource_limit: f64,
    pub jobs: Vec<Rc<Job>>,
    pub p_max: f64,
}
impl ProblemData {
    fn from(instance: Instance) -> Self {
        let Instance {
            epsilon,
            machine_count,
            resource_limit,
            jobs,
        } = instance;
        let epsilon_prime = 0.2 * epsilon; // epsilon / 5
        let p_max = jobs
            .iter()
            .max_by(|job0, job1| {
                job0.processing_time
                    .partial_cmp(&job1.processing_time)
                    .expect("invalid processing time")
            })
            .expect("no jobs found")
            .processing_time;

        let mut job_ids = 0..;
        ProblemData {
            epsilon,
            epsilon_squared: epsilon * epsilon,
            epsilon_prime,
            epsilon_prime_squared: epsilon_prime * epsilon_prime,
            one_over_epsilon_prime: (1. / epsilon_prime) as i32,
            machine_count,
            resource_limit,
            jobs: jobs
                .into_iter()
                .map(|job| {
                    Rc::new(Job {
                        id: job_ids.next().unwrap(),
                        processing_time: job.processing_time,
                        resource_amount: job.resource_amount,
                    })
                })
                .collect(),
            p_max,
        }
    }
    fn is_wide(&self, job: &Rc<Job>) -> bool {
        job.resource_amount >= self.epsilon_prime * self.resource_limit
    }
}

#[derive(Debug, Clone)]
pub struct Job {
    pub id: i32,
    pub processing_time: f64,
    pub resource_amount: f64,
}
impl PartialOrd for Job {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.resource_amount.partial_cmp(&other.resource_amount)
    }
}
impl Ord for Job {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other)
            .expect("invalid resource amount, cannot compare jobs")
    }
}

impl PartialEq for Job {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Job {}

#[derive(Debug)]
pub struct Schedule {
    pub mapping: Box<Vec<JobPosition>>,
}

#[derive(Debug)]
pub struct JobPosition {
    pub machine: i32,
    pub starting_time: f64,
}

pub fn compute_schedule(instance: Instance) -> Schedule {
    println!("Computing schedule");
    let Instance {
        epsilon,
        machine_count,
        ..
    } = instance;

    if 1.0 / epsilon >= machine_count.into() {
        todo!("second case");
    }

    let problem_data = ProblemData::from(instance);
    // let (wide_jobs, narrow_jobs): (Vec<Job>, Vec<Job>) = {
    //     let (mut wide_jobs, narrow_jobs) = problem_data
    //         .clone()
    //         .jobs
    //         .into_iter()
    //         .partition::<Vec<_>, _>(|job| problem_data.is_wide(job));
    //     wide_jobs.sort();
    //     (wide_jobs, narrow_jobs)
    // };

    // println!("Wide {:?}", wide_jobs);
    // println!("Narrow {:?}", narrow_jobs);

    // let _i_sup = create_i_sup(wide_jobs, &problem_data);

    let _ = max_min(problem_data);

    Schedule {
        mapping: Box::from(vec![]),
    }
}

fn create_i_sup(wide_jobs: Vec<Job>, problem_data: &ProblemData) -> Vec<Job> {
    println!("Computing I_sup from {} wide jobs", wide_jobs.len());
    let ProblemData {
        epsilon_prime_squared,
        ..
    } = problem_data;
    let p_w: f64 = wide_jobs.iter().map(|job| job.processing_time).sum();
    let step = epsilon_prime_squared * p_w;
    let mut job_ids = (wide_jobs.last().expect("last job").id + 1)..;
    let groups = linear_grouping(step, &wide_jobs);
    let additional_jobs = groups
        .into_iter()
        .map(|group| {
            let resource_amount = group
                .into_iter()
                .max()
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
        "Obtained I_sup with {} additional jobs, totalling {} jobs",
        additional_jobs.len(),
        wide_jobs.len() + additional_jobs.len(),
    );
    [wide_jobs, additional_jobs].concat()
}

fn linear_grouping(step: f64, jobs: &Vec<Job>) -> Vec<Vec<Job>> {
    let n = jobs.len();
    println!("Grouping {} jobs", n);
    // FIXME: Add special handling for the last group since we already know that
    // all remaining jobs will be put into it. Due to floating point
    // imprecision, it might happen that we accidentally open one group to many,
    // containing a single job only, having the size of the floating point
    // rounding error.
    if n == 0 {
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
    println!("Obtained {} groups", groups.len());
    groups
}

#[derive(Debug, Clone)]
struct Configuration {
    jobs: Vec<(Rc<Job>, i32)>, // job.id -> how many times it is contained
    processing_time: f64,
    resource_amount: f64,
    machine_count: i32,
}
impl Configuration {
    fn new(jobs: &Vec<(Rc<Job>, i32)>) -> Self {
        Configuration {
            jobs: jobs.to_vec(),
            processing_time: jobs
                .iter()
                .map(|(job, count)| *count as f64 * job.processing_time)
                .sum(),
            resource_amount: jobs
                .iter()
                .map(|(job, count)| *count as f64 * job.resource_amount)
                .sum(),
            machine_count: jobs.iter().map(|&(_, count)| count).sum(),
        }
    }
    // fn get(&self, job: &Job) -> Option<i32> {
    //     Some(*self.jobs.get(&job.id)?)
    // }
    // fn can_add(&self, job: &Job, problem: &ProblemData) -> bool {
    //     self.get(job).unwrap_or(0) + 1
    //         <= if problem.is_wide(job) {
    //             problem.one_over_epsilon_prime
    //         } else {
    //             1
    //         }
    //         && self.machine_count + 1 <= problem.machine_count
    //         && self.resource_amount + job.resource_amount <= problem.resource_limit
    // }
    // fn set(&mut self, job: Job, count: i32) {
    //     match self.index.get(&job.id) {
    //         None => {
    //             let i = self.jobs.len();
    //             self.index.insert(job.id, i);
    //             self.jobs.push((job, count));
    //         }
    //         Some(i) => {
    //             self.jobs[*i] = (job, count);
    //         }
    //     }
    // }
    // fn is_valid(&self, instance: Instance) -> bool {
    //     self.machines() <= instance.machine_count
    //         && self.resource_amount() <= instance.resource_limit
    // }
}
impl Hash for Configuration {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (job, count) in self.jobs.iter() {
            job.id.hash(state);
            count.hash(state);
        }
    }
}

// fn enumerate_all_configurations<'a>(
//     problem: &'a ProblemData,
// ) -> Box<dyn Iterator<Item = Configuration> + 'a> {
//     // get empty config
//     let config = Configuration::empty();
//     // make recursive call
//     search_configurations(config, problem, 0)
// }

// fn search_configurations<'a>(
//     config: Configuration,
//     problem: &'a ProblemData,
//     skip: usize,
// ) -> Box<dyn Iterator<Item = Configuration> + 'a> {
//     let iterator = problem
//         .jobs
//         .iter()
//         .skip(skip)
//         .filter_map(move |job| {
//             if config.can_add(job, problem) {
//                 Some(Configuration::from(&config, job))
//             } else {
//                 None
//             }
//         })
//         .enumerate()
//         .flat_map(move |(i, c)| once(c.clone()).chain(search_configurations(c, problem, skip + i)));
//     Box::new(iterator) as Box<dyn Iterator<Item = Configuration> + 'a>
// }

// fn index_configurations(
//     configurations: Vec<Rc<Configuration>>,
// ) -> HashMap<i32, Vec<Rc<Configuration>>> {
//     let mut index = HashMap::new();
//     for config in configurations.into_iter() {
//         for &job_id in config.jobs.keys() {
//             match index.get_mut(&job_id) {
//                 None => {
//                     index.insert(job_id, vec![Rc::clone(&config)]);
//                 }
//                 Some(vec) => {
//                     vec.push(Rc::clone(&config));
//                 }
//             }
//         }
//     }
//     index
// }

// fn f(x: Vec<i32>, j: Job, c_i: Vec<Rc<Configuration>>) -> f64 {
//     c_i.into_iter().map(|c| {
//         let c_j = c.jobs.get(&j.id).unwrap_or(&0);
//         let x_c = c.processing_time * x[c];
//     });
//     0.0
// }

fn unit(i: usize, m: usize) -> Vec<f64> {
    let mut temp_vec = vec![0.0f64; m];
    temp_vec[i] = 1.0f64;
    temp_vec
}

fn max_min(problem_data: ProblemData) -> Vec<Configuration> {
    println!("Solving max-min");
    let ProblemData {
        epsilon_squared,
        epsilon_prime,
        ref jobs,
        ..
    } = problem_data;
    // compute initial solution;
    let _rho = epsilon_prime / (1.0 + epsilon_prime);
    println!("Computing initial solution");
    let m = jobs.len();
    let scale = 1.0f64 / (m as f64);
    let units = (0..m).map(|i| unit(i, m));
    // job identifier -> number of times included in configuration -> how many times was it picked
    // (job.id -> C(j)) -> x_c
    let mut x: HashMap<Configuration, f64>;
    let mut solution: Vec<_> = units
        .map(|e| solve_block_problem_ilp(e, 0.5, &problem_data))
        .collect();
    // .fold(vec![0.0; m], |acc: Vec<f64>, x: Vec<f64>| {
    //     // vec add
    //     acc.iter().zip(x).map(|(x, y)| x + y).collect::<Vec<f64>>()
    // });

    // let configurations: Vec<_> = enumerate_all_configurations(&problem_data).collect();
    // let index = index_configurations(configurations);
    let fx = vec![]; // TODO: continue

    // iterate
    loop {
        // price vector
        let prec = epsilon_squared / (m as f64);
        let theta = find_theta(epsilon_prime, &fx, prec);
        let price = compute_price(&fx, epsilon_prime, theta);
        println!("++ Starting iteration with price {:?}", price);
        // solve block problem
        let max = solve_block_problem_ilp(price, epsilon_prime, &problem_data);
        println!("Received block problem solution {:?}", max);
        // update solution = ((1-tau) * solution) + (tau * solution)
        let tau = compute_step_length();
        let one_minus_tau = 1.0 - tau;
        for i in 0..jobs.len() {
            // solution[i] = one_minus_tau * solution[i] + tau * solution[i]
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

// job: 0 1 2
// vec: 1 0 5
// -> config: (0: 1), (2, 5)

// con: 0 1 2
// x  : 1 0 5
// x_c mit c=2 => 5
// ->   5

fn solve_block_problem_ilp(
    q: Vec<f64>,
    precision: f64,
    problem_data: &ProblemData,
) -> Configuration {
    let ProblemData {
        machine_count,
        resource_limit,
        jobs,
        ..
    } = problem_data;
    let wide_job_max_count = (1.0 / precision) as i32;
    let mut prob = Ilp::new(*machine_count, *resource_limit);

    let variables: Vec<(Rc<Job>, Variable)> = jobs
        .iter()
        .zip(q.to_vec())
        .map(|(job, q)| {
            let job = Rc::clone(job);
            let c = ConfigurationCandidate {
                p: job.processing_time,
                r: job.resource_amount,
                max_a: if problem_data.is_wide(&job) {
                    wide_job_max_count
                } else {
                    1
                },
                q,
            };
            // println!("Adding variable {:?}", c);
            let var = prob.add(c);
            (job, var)
        })
        .collect();

    let solution = prob.find_configuration();
    let a_star: Vec<(Rc<Job>, i32)> = variables
        .into_iter()
        .map(|(job, var)| (job, solution.value(var) as i32))
        .filter(|(_, var)| var != 0)
        .collect();
    println!("Solved block problem ILP (n={}) for {:?}", jobs.len(), q);
    Configuration::new(&a_star)
}

// find a theta binary search using 2.3
// 2.3
fn find_theta(t: f64, fx: &Vec<f64>, prec: f64) -> f64 {
    let mut upper = *fx
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
        .expect("no min found while searching theta");
    let mut lower = 0.0;
    let mut act = (upper + lower) / 2.0;

    while (act - (upper + lower) / 2.0).abs() > prec {
        act = (upper + lower) / 2.0;
        let val = compute_theta_f(t, act, fx);

        if (val - 1.0).abs() < prec {
            break;
        } else if val - 1.0 < 0.0 {
            lower = act;
        } else {
            upper = act;
        }
    }
    act
}

fn compute_theta_f(t: f64, theta: f64, fx: &Vec<f64>) -> f64 {
    let sum: f64 = fx.iter().map(|x| theta / (*x - theta)).sum();
    sum * (t / fx.len() as f64)
}

fn compute_price(fx: &Vec<f64>, t: f64, theta: f64) -> Vec<f64> {
    let r = t / fx.len() as f64;
    fx.iter().map(|x| r * theta / (*x - theta)).collect()
}

fn compute_step_length() -> f64 {
    0.0
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
