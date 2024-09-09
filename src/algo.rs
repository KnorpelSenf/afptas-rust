use good_lp::{
    constraint, default_solver, variable, variables, Expression, ProblemVariables, Solution,
    SolverModel, Variable,
};
use std::{
    cmp::{max, min, Ordering},
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::{Hash, Hasher},
    iter::repeat,
    vec,
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
struct ProblemData {
    epsilon: f64,
    epsilon_squared: f64,
    epsilon_prime: f64,
    epsilon_prime_squared: f64,
    one_over_epsilon_prime: i32,
    machine_count: i32,
    machine_count_usize: usize,
    resource_limit: f64,
    jobs: Vec<Job>,
    p_max: f64,

    epsilon_prime_times_resource_limit: f64,
    second_case: bool,
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
            machine_count_usize: machine_count as usize,
            resource_limit,
            jobs: jobs
                .into_iter()
                .map(|job| Job {
                    id: job_ids.next().unwrap(),
                    processing_time: job.processing_time,
                    resource_amount: job.resource_amount,
                })
                .collect(),
            p_max,

            epsilon_prime_times_resource_limit: epsilon_prime * resource_limit,
            second_case: 1.0 / epsilon >= machine_count.into(),
        }
    }
    fn is_wide(&self, job: &Job) -> bool {
        self.second_case || job.resource_amount >= self.epsilon_prime_times_resource_limit
    }
}

#[derive(Copy, Clone)]
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
impl Debug for Job {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id = self.id;
        f.write_str(&format!("J{id}"))
    }
}
impl Hash for Job {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Clone, Debug)]
pub struct Schedule {
    pub mapping: Vec<MachineSchedule>,
}
impl Schedule {
    fn empty(machine_count: usize) -> Self {
        Schedule {
            mapping: vec![MachineSchedule::empty(); machine_count],
        }
    }
    fn add(&mut self, machine: usize, job: Job) {
        self.mapping[machine].push(job)
    }
}

#[derive(Clone, Debug)]
pub struct MachineSchedule {
    pub jobs: Vec<Job>,
}
impl MachineSchedule {
    fn empty() -> Self {
        MachineSchedule { jobs: vec![] }
    }
    fn push(&mut self, job: Job) {
        self.jobs.push(job)
    }
}

pub fn compute_schedule(instance: Instance) -> Schedule {
    let problem_data = ProblemData::from(instance);
    let (wide_jobs, narrow_jobs): (Vec<Job>, Vec<Job>) = problem_data
        .jobs
        .iter()
        .partition(|job| problem_data.is_wide(job));
    println!(
        "Computing schedule from {} wide and {} narrow jobs",
        wide_jobs.len(),
        narrow_jobs.len()
    );

    let job_len = problem_data.jobs.len();
    let x = max_min(&problem_data);
    println!("Max-min solved with:");
    print_selection(job_len, problem_data.machine_count_usize, &x);
    let x = reduce_to_basic_solution(x);
    println!("Reduced the max-min solution min to:");
    print_selection(job_len, problem_data.machine_count_usize, &x);
    let (x_tilde, y_tilde) = generalize(&problem_data, x);
    println!("Generalized to:");
    print_gen_selection(
        job_len,
        problem_data.machine_count_usize,
        problem_data.resource_limit,
        &x_tilde,
    );
    let (x_bar, y_bar) = reduce_resource_amounts(&problem_data, &x_tilde, &y_tilde);
    // println!("x_bar entries:");
    // println!("{:#?}", x_bar.configurations);
    // println!("y_bar entries are:");
    // println!("{:#?}", y_bar.processing_times);

    integral_schedule(&problem_data, x_bar, y_bar)
}

// fn create_i_sup(wide_jobs: Vec<Job>, problem_data: &ProblemData) -> (Vec<Vec<Job>>, Vec<Job>) {
//     println!("Computing I_sup from {} wide jobs", wide_jobs.len());
//     let ProblemData {
//         epsilon_prime_squared,
//         ..
//     } = problem_data;
//     let p_w: f64 = wide_jobs.iter().map(|job| job.processing_time).sum();
//     let step = epsilon_prime_squared * p_w;
//     let mut job_ids = (wide_jobs.last().expect("no jobs").id + 1)..;
//     let groups = linear_grouping(step, &wide_jobs);
//     let additional_jobs = groups
//         .iter()
//         .map(|group| {
//             let resource_amount = group.iter().max().expect("empty group").resource_amount;
//             Job {
//                 id: job_ids.next().unwrap(),
//                 processing_time: step,
//                 resource_amount,
//             }
//         })
//         .collect::<Vec<_>>();
//     println!(
//         "Obtained I_sup with {} additional jobs, totalling {} jobs",
//         additional_jobs.len(),
//         wide_jobs.len() + additional_jobs.len(),
//     );
//     (groups, [wide_jobs, additional_jobs].concat())
// }

// fn linear_grouping(step: f64, jobs: &Vec<Job>) -> Vec<Vec<Job>> {
//     let n = jobs.len();
//     println!("Grouping {} jobs: {jobs:?}", n);
//     // FIXME: Add special handling for the last group since we already know that
//     // all remaining jobs will be put into it. Due to floating point
//     // imprecision, it might happen that we accidentally open one group to many,
//     // containing a single job only, having the size of the floating point
//     // rounding error.
//     if n == 0 {
//         return vec![];
//     }
//     let mut job_ids = 0..;

//     let mut groups: Vec<Vec<Job>> = vec![];
//     let mut current_group: Vec<Job> = vec![];
//     let mut current_processing_time = 0.0f64;
//     for job in jobs.iter() {
//         let mut remaining_processing_time = job.processing_time;
//         loop {
//             let remaining_space = (groups.len() + 1) as f64 * step - current_processing_time;
//             // Handle last iteration if the job fits entirely
//             if remaining_processing_time <= remaining_space {
//                 let new_job = Job {
//                     id: job_ids.next().unwrap(),
//                     processing_time: remaining_processing_time,
//                     resource_amount: job.resource_amount,
//                 };
//                 current_processing_time += remaining_processing_time;
//                 current_group.push(new_job);
//                 break;
//             }

//             // Split off a bit of the job for the current group
//             let new_job = Job {
//                 id: job_ids.next().unwrap(),
//                 processing_time: remaining_space,
//                 resource_amount: job.resource_amount,
//             };
//             current_group.push(new_job);
//             groups.push(current_group);

//             current_group = vec![];

//             current_processing_time += remaining_space;
//             remaining_processing_time -= remaining_space;
//         }
//     }
//     println!("Obtained {} groups", groups.len());
//     groups
// }

#[derive(Clone)]
struct Configuration {
    /// job -> index in vector
    index: HashMap<Job, usize>,
    /// job -> how many times it is contained
    jobs: Vec<(Job, i32)>,
    /// precomputed processing time
    processing_time: f64,
    /// precomputed resource amount
    resource_amount: f64,
    /// precomputed machine count
    machine_count: i32,
}
impl Debug for Configuration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Config")?;
        f.debug_list().entries(&self.jobs).finish()?;
        Ok(())
    }
}
impl Configuration {
    fn new(jobs: Vec<(Job, i32)>) -> Self {
        if jobs.windows(2).any(|pair| pair[0].0.id >= pair[1].0.id) {
            panic!("jobs are out of order");
        }
        let (processing_time, resource_amount, machine_count) =
            jobs.iter().fold((0.0, 0.0, 0), |(p, r, m), (job, count)| {
                (
                    job.processing_time.max(p),
                    r + job.resource_amount * *count as f64,
                    m + *count,
                )
            });
        let index = HashMap::from_iter(jobs.iter().enumerate().map(|(i, &(job, _))| (job, i)));
        Configuration {
            jobs,
            index,
            processing_time,
            resource_amount,
            machine_count,
        }
    }
    /// C(j)
    fn job_count(&self, job: &Job) -> i32 {
        match self.index.get(job) {
            None => 0,
            Some(i) => self.jobs.get(*i).map_or(0, |pair| pair.1),
        }
    }

    fn reduce_to_wide_jobs(&self, problem: &ProblemData) -> Configuration {
        Configuration::new(
            self.jobs
                .iter()
                .filter(|(job, _)| problem.is_wide(job))
                .map(|(job, count)| (job.clone(), *count))
                .collect(),
        )
    }
}
impl PartialEq for Configuration {
    fn eq(&self, other: &Self) -> bool {
        self.jobs.eq(&other.jobs)
    }
}
impl Eq for Configuration {}
impl Hash for Configuration {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (job, count) in self.jobs.iter() {
            job.id.hash(state);
            count.hash(state);
        }
    }
}

#[derive(Debug)]
struct Selection(HashMap<Configuration, f64>);
impl Selection {
    fn single(config: Configuration) -> Self {
        let mut h = HashMap::new();
        h.insert(config, 1.0);
        Selection(h)
    }
    fn init(iter: impl IntoIterator<Item = (Configuration, f64)>) -> Self {
        Selection(HashMap::from_iter(iter))
    }

    fn interpolate(self: &mut Self, s: Selection, tau: f64) {
        assert!(
            s.0.len() == 1,
            "Newly chosen selection must contain exactly one configuration"
        );
        self.scale(1.0 - tau);
        s.0.into_iter().for_each(|(c, v)| {
            let old = self.0.get(&c).unwrap_or(&0.0);
            self.0.insert(c, v * tau + old);
        });
    }
    fn scale(self: &mut Self, factor: f64) {
        self.0.values_mut().for_each(|v| *v *= factor);
    }
}

fn f(j: &Job, x: &Selection) -> f64 {
    x.0.iter()
        .map(|(c, x_c)| c.job_count(j) as f64 * x_c)
        .sum::<f64>()
        / j.processing_time
}

fn unit(i: usize, m: usize) -> Vec<f64> {
    let mut temp_vec = vec![0.0; m];
    temp_vec[i] = 1.0;
    temp_vec
}

fn max_min(problem_data: &ProblemData) -> Selection {
    println!("Solving max-min");
    let ProblemData {
        epsilon,
        epsilon_squared,
        epsilon_prime,
        ref jobs,
        ..
    } = problem_data;
    // compute initial solution;
    println!("Computing initial solution");
    let m = jobs.len();
    let mut x = Selection::init((0..m).map(|i| {
        (
            solve_block_problem_ilp(&unit(i, m), 0.5, &problem_data),
            1.0 / m as f64,
        )
    }));
    println!("Initial value is {x:?}");

    // iterate
    loop {
        let fx: Vec<f64> = jobs.iter().map(|job| f(job, &x)).collect();
        println!("f(x) = {fx:?}");
        // price vector
        let prec = epsilon_squared / (m as f64);
        let theta = find_theta(*epsilon_prime, &fx, prec);
        let price = compute_price(&fx, *epsilon_prime, theta);
        println!("++ Starting iteration with price {price:?}");
        // solve block problem
        let config = solve_block_problem_ilp(&price, *epsilon_prime, &problem_data);
        let y = Selection::single(config);
        println!("Received block problem solution {y:?}");
        let fy: Vec<f64> = jobs.iter().map(|job| f(job, &y)).collect();
        println!("f(y) = {fy:?}");

        // compute v
        let v = compute_v(&price, &fx, &fy);
        println!("v = {v}");
        if v < *epsilon_prime {
            let λ_hat = fx
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .unwrap_or(&1.0);
            println!("λ^ = {}", λ_hat);
            x.scale(1.0 / λ_hat);
            break;
        }
        // update solution = ((1-tau) * solution) + (tau * solution)
        let tau = line_search(&fx, &fy, theta, *epsilon_prime, *epsilon);
        // let one_minus_tau = 1.0 - tau;
        // for i in 0..jobs.len() {
        // solution[i] = one_minus_tau * solution[i] + tau * solution[i]
        // }
        x.interpolate(y, tau);
        println!(
            "Updated solution with step length tau={} to be {:?}",
            tau, x
        );
    }
    x
}
fn print_selection(job_len: usize, m: usize, x: &Selection) {
    let digits_per_job_id = (job_len - 1).to_string().len();
    let lcol = max(4, digits_per_job_id * m);
    println!("{: >lcol$} | Length", "Jobs");
    println!("{:->lcol$}---{}", "-", "-".repeat(19));
    for (c, x_c) in x.0.iter() {
        let job_ids = c
            .jobs
            .iter()
            .map(|job| {
                format!("{: >digits_per_job_id$}", job.0.id.to_string()).repeat(job.1 as usize)
            })
            .collect::<Vec<_>>()
            .join("");
        println!("{: >lcol$} | {}", job_ids, x_c);
    }
}
fn print_gen_selection(job_len: usize, m: usize, r: f64, x: &GeneralizedSelection) {
    let digits_per_job_id = (job_len - 1).to_string().len();
    let digits_per_machine = m.to_string().len();
    let resource_precision = 2;
    let digits_per_resource = r.ceil().to_string().len() + 1 + resource_precision;
    let lcol = max("Jobs".len(), digits_per_job_id * m);
    let mcol = max(
        "w=(m,r)".len(),
        1 + digits_per_machine + ", ".len() + digits_per_resource + 1,
    );
    println!("{:>lcol$} | {:<mcol$} | Length", "Jobs", "w=(m,r)",);
    println!("{:->lcol$}---{:-<mcol$}---{}", "-", "-", "-".repeat(19));
    for (c, x_c) in x.configurations.iter() {
        let job_ids = c
            .configuration
            .jobs
            .iter()
            .map(|job| {
                format!("{:>digits_per_job_id$}", job.0.id.to_string()).repeat(job.1 as usize)
            })
            .collect::<Vec<_>>()
            .join("");
        let win = format!(
            "({:>digits_per_machine$}, {:>digits_per_resource$})",
            c.window.machine_count,
            format!("{:.resource_precision$}", c.window.resource_amount)
        );
        println!("{:>lcol$} | {win} | {}", job_ids, x_c);
    }
}

fn solve_block_problem_ilp(
    q: &Vec<f64>,
    precision: f64,
    problem_data: &ProblemData,
) -> Configuration {
    let ProblemData {
        machine_count,
        resource_limit,
        ref jobs,
        ..
    } = *problem_data;
    let wide_job_max_count = (1.0 / precision) as i32;
    let mut prob = Ilp::new(machine_count, resource_limit);

    let variables: Vec<(Job, Variable)> = jobs
        .iter()
        .zip(q.iter())
        .filter(|(_, &q)| q > 0.0)
        .map(|(&job, &q)| {
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
            let var = prob.add(c);
            (job, var)
        })
        .collect();

    let solution = prob.find_configuration();

    let a_star: Vec<(Job, i32)> = variables
        .into_iter()
        .map(|(job, var)| (job, solution.value(var) as i32))
        .filter(|(_, var)| *var != 0)
        .collect();
    println!("Solved ILP for {} items with {:?}", q.len(), a_star);
    Configuration::new(a_star)
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

fn compute_v(p: &Vec<f64>, fx: &Vec<f64>, fy: &Vec<f64>) -> f64 {
    let a = scalar_product(p, fy);
    let b = scalar_product(p, fx);
    (a - b) / (a + b)
}

fn scalar_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn line_search(fx: &Vec<f64>, fy: &Vec<f64>, theta: f64, t: f64, epsilon: f64) -> f64 {
    let mut up = 1.0;
    let mut low = 0.0;

    // perform a binary search
    while low < (1.0 - epsilon) * up {
        let act = (up + low) / 2.0;

        // test if the potential function is still defined
        let defined = fx
            .iter()
            .zip(fy.iter())
            .all(|(&x, &y)| x + act * (y - x) > theta);

        if !defined {
            up = act;
        } else {
            let val = derivative_pot(act, fx, fy, t, theta);
            if val > 0.0 {
                low = act;
            } else {
                up = act;
            }
        }
    }
    (low + up) / 2.0
}

fn derivative_pot(tau: f64, fx: &Vec<f64>, fy: &Vec<f64>, t: f64, theta: f64) -> f64 {
    let res: f64 = fx
        .iter()
        .zip(fy.iter())
        .map(|(&x, &y)| (y - x) / (x + tau * (y - x) - theta))
        .sum();
    res * t / fx.len() as f64
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

fn reduce_to_basic_solution(x: Selection) -> Selection {
    println!("Reducing to basic solution");
    // TODO: implement
    x
}

fn generalize(problem: &ProblemData, x: Selection) -> (GeneralizedSelection, NarrowJobSelection) {
    let narrow_jobs: Vec<Job> = problem
        .jobs
        .iter()
        .filter(|job| !problem.is_wide(job))
        .copied()
        .collect();
    let (x_tilde, y_tilde) =
        x.0.iter()
            .map(|(c, x_c)| (c, c.reduce_to_wide_jobs(problem), x_c))
            .fold(
                (HashMap::new(), NarrowJobSelection::empty()),
                |(mut acc_x, mut acc_y), (c, c_w, x_c)| {
                    let window = Window::main(problem, &c_w);

                    for narrow_job in narrow_jobs.iter() {
                        acc_y.add(
                            NarrowJobConfiguration {
                                narrow_job: *narrow_job,
                                window,
                            },
                            c.job_count(narrow_job) as f64 * x_c,
                        )
                    }

                    let gen = GeneralizedConfiguration {
                        configuration: c_w,
                        window,
                    };
                    acc_x
                        .entry(gen)
                        .and_modify(|existing| *existing += *x_c)
                        .or_insert(*x_c);

                    (acc_x, acc_y)
                },
            );

    (GeneralizedSelection::from(x_tilde), y_tilde)
}

#[derive(Copy, Clone)]
struct Window {
    resource_amount: f64,
    machine_count: i32,
}
impl Window {
    fn empty() -> Self {
        Window {
            resource_amount: 0.0,
            machine_count: 0,
        }
    }
    fn main(problem: &ProblemData, config: &Configuration) -> Self {
        let machine_count = problem.machine_count - config.machine_count;
        let resource_amount = if machine_count == 0 {
            0.0
        } else {
            problem.resource_limit - config.resource_amount
        };
        Window {
            resource_amount,
            machine_count,
        }
    }
}
impl PartialEq for Window {
    fn eq(&self, other: &Self) -> bool {
        self.resource_amount.eq(&other.resource_amount) && self.machine_count == other.machine_count
    }
}
impl Eq for Window {}
impl Hash for Window {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.resource_amount.to_bits().hash(state);
        self.machine_count.hash(state);
    }
}
impl Debug for Window {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[R=")?;
        f.write_str(&self.resource_amount.to_string())?;
        f.write_str(",m=")?;
        f.write_str(&self.machine_count.to_string())?;
        f.write_str("]")?;
        Ok(())
    }
}

#[derive(Clone)]
struct GeneralizedConfiguration {
    configuration: Configuration,
    window: Window,
}
impl PartialEq for GeneralizedConfiguration {
    fn eq(&self, other: &Self) -> bool {
        self.configuration.eq(&other.configuration)
    }
}
impl Eq for GeneralizedConfiguration {}
impl Hash for GeneralizedConfiguration {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.configuration.hash(state)
    }
}
impl Debug for GeneralizedConfiguration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("GConfig[{:.3}]", self.configuration.processing_time).as_str())?;
        self.window.fmt(f)?;
        f.debug_list().entries(&self.configuration.jobs).finish()?;
        Ok(())
    }
}
#[derive(Clone, Debug)]
struct GeneralizedSelection {
    configurations: Vec<(GeneralizedConfiguration, f64)>,
}
impl GeneralizedSelection {
    fn empty() -> Self {
        GeneralizedSelection {
            configurations: vec![],
        }
    }
    fn from(mapping: HashMap<GeneralizedConfiguration, f64>) -> Self {
        GeneralizedSelection {
            configurations: mapping.into_iter().collect(),
        }
    }
    fn merge(selections: Vec<GeneralizedSelection>) -> Self {
        GeneralizedSelection {
            configurations: selections
                .into_iter()
                .flat_map(|s| s.configurations)
                .collect(),
        }
    }
    fn push(&mut self, config: GeneralizedConfiguration, x_c: f64) {
        self.configurations.push((config, x_c));
    }
    fn sort_by_resource_amount(&mut self) {
        self.configurations.sort_by(|left, right| {
            left.0
                .configuration
                .resource_amount
                .partial_cmp(&right.0.configuration.resource_amount)
                .expect("cannot compare resource amounts")
        })
    }
}

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
struct NarrowJobConfiguration {
    narrow_job: Job,
    window: Window,
}
impl Debug for NarrowJobConfiguration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NConfig")?;
        self.window.fmt(f)?;
        self.narrow_job.fmt(f)?;
        Ok(())
    }
}
#[derive(Debug)]
struct NarrowJobSelection {
    processing_times: HashMap<NarrowJobConfiguration, f64>,
    windex: HashMap<Window, HashSet<Job>>,
}
impl NarrowJobSelection {
    fn get_jobs_by_window(&self, window: &Window) -> Vec<(Job, f64)> {
        let set = self.windex.get(window);
        let window = *window;
        if let Some(jobs) = set {
            jobs.iter()
                .copied()
                .map(|job| {
                    let config = NarrowJobConfiguration {
                        narrow_job: job,
                        window,
                    };
                    let processing_time = *self.processing_times.get(&config).expect(
                        "invalid internal state of narrow jobs selection, index out of date",
                    );
                    (job, processing_time)
                })
                .collect()
        } else {
            vec![]
        }
    }
    fn add(&mut self, config: NarrowJobConfiguration, processing_time: f64) {
        self.processing_times
            .entry(config)
            .and_modify(|old| *old += processing_time)
            .or_insert(processing_time);
        self.windex
            .entry(config.window)
            .and_modify(|set| {
                set.insert(config.narrow_job);
            })
            .or_insert_with(|| {
                let mut s = HashSet::new();
                s.insert(config.narrow_job);
                s
            });
    }

    fn empty() -> Self {
        NarrowJobSelection {
            processing_times: HashMap::new(),
            windex: HashMap::new(),
        }
    }
    fn merge(selections: Vec<NarrowJobSelection>) -> Self {
        selections
            .into_iter()
            .fold(NarrowJobSelection::empty(), |merged, sel| {
                sel.processing_times.into_iter().fold(
                    merged,
                    |mut acc, (config, processing_time)| {
                        acc.add(config, processing_time);
                        acc
                    },
                )
            })
    }
}

fn reduce_resource_amounts(
    problem: &ProblemData,
    x_tilde: &GeneralizedSelection,
    y_tilde: &NarrowJobSelection,
) -> (GeneralizedSelection, NarrowJobSelection) {
    println!("Reducing resource amounts");
    let (p_pre, k) = group_by_machine_count(problem, x_tilde);

    println!("p_pre={p_pre}");
    let step_width = problem.epsilon_prime_squared * p_pre;
    println!("Step width is {step_width}");

    let (stacks, narrow_jobs): (Vec<GeneralizedSelection>, Vec<NarrowJobSelection>) = k
        .into_iter()
        .map(|(x_c, k_i)| {
            println!("  Processing generalized selection with x_c={x_c}");
            let (sel, narrow, _, _, _) = k_i
                .configurations
                .into_iter()
                // fold over configs in this stack
                // - generalized selection that is a vector of all the configuration snippets
                // - narrow job selection that is a vector of all the narrow jobs
                // - processing time: remaining distance to the bottom of the stack
                // - window: current window size to be added to each config
                // - k_ik that collects the configurations up to each cut in order to compute phi_c
                .fold(
                    (
                        GeneralizedSelection::empty(),
                        NarrowJobSelection::empty(),
                        x_c,
                        Window::empty(),
                        vec![],
                    ),
                    |(mut wide_sel, mut narrow_sel, e_c, mut window, mut k_i), (c, mut p)| {
                        println!("    Processing configuration {:?} with p={p}", c);
                        let s_c = e_c - p;
                        let cur_step = (e_c / step_width).floor();
                        let next_step = (s_c / step_width).floor();
                        let is_cut = cur_step != next_step;

                        println!("    s_c={s_c} --- {e_c}=e_c");
                        print!("    current={cur_step} --- {next_step}=next");
                        if is_cut {
                            println!(" ++ CUT! ++");
                        } else {
                            println!();
                        }

                        k_i.push(c.clone());

                        if is_cut {
                            let lowest_cut = (s_c / step_width).ceil() * step_width;
                            println!("    lowest cut={lowest_cut}");
                            let p_w_ik: f64 = k_i
                                .iter()
                                .map(|k_ik| {
                                    println!(
                                        "    Adding narrow jobs from K_ik by window {:?}",
                                        k_ik.window
                                    );
                                    y_tilde.get_jobs_by_window(&k_ik.window).iter().for_each(
                                        |(narrow_job, amount)| {
                                            println!(
                                                "      Adding {:?} with amount={amount}",
                                                narrow_job
                                            );
                                            narrow_sel.add(
                                                NarrowJobConfiguration {
                                                    narrow_job: *narrow_job,
                                                    window,
                                                },
                                                *amount,
                                            );
                                        },
                                    );

                                    k_ik.configuration.processing_time
                                })
                                .sum();
                            k_i = vec![];

                            let w_down = e_c - lowest_cut;
                            let phi_down = w_down / p_w_ik;
                            let phi_up = 1.0 - phi_down;
                            let next_window = c.window;
                            println!("    Next window will be {:?}", c.window);

                            println!("    Adding narrow jobs by window {:?}", window);
                            y_tilde.get_jobs_by_window(&window).iter().for_each(
                                |(narrow_job, amount)| {
                                    println!(
                                        "      Adding {:?} with amount={} to current window",
                                        narrow_job,
                                        *amount * phi_up
                                    );
                                    narrow_sel.add(
                                        NarrowJobConfiguration {
                                            narrow_job: *narrow_job,
                                            window,
                                        },
                                        *amount * phi_up,
                                    );
                                    // TODO: this also has to happen for the last window (R,m), which should be
                                    // x_bar(emptyset, (R,m)) = x_tilde(emptyset, (R,m)) + epsilon_prime * P_pre
                                    println!(
                                        "      Adding {:?} with amount={} to next window",
                                        narrow_job,
                                        *amount * phi_down
                                    );
                                    narrow_sel.add(
                                        NarrowJobConfiguration {
                                            narrow_job: *narrow_job,
                                            window: next_window,
                                        },
                                        *amount * phi_down,
                                    );
                                },
                            );

                            let highest_cut = cur_step * step_width;
                            let p_diff = e_c - highest_cut;
                            println!("    Adding generalized config with x_c={p_diff}");
                            wide_sel.push(
                                GeneralizedConfiguration {
                                    configuration: c.configuration.clone(),
                                    window,
                                },
                                p_diff,
                            );

                            window = next_window;
                            p -= p_diff;
                        }

                        println!("    Adding generalized config with x_c={p}");
                        wide_sel.push(
                            GeneralizedConfiguration {
                                configuration: c.configuration,
                                window,
                            },
                            p,
                        );

                        println!("    Done processing configuration, p is now {p}");
                        (wide_sel, narrow_sel, s_c, window, k_i)
                    },
                );
            println!(
                "  Obtained a selection with {} wide jobs and a selection with {} narrow jobs",
                sel.configurations.len(),
                narrow.processing_times.len()
            );
            println!("  Done processing generalized selection with x_c={x_c}");
            (sel, narrow)
        })
        .unzip();

    println!("Done creating {} stacks:", stacks.len());
    for (i, stack) in stacks.iter().enumerate() {
        println!("  --- K_{} ---  ", i + 1);
        for (config, x_c) in stack.configurations.iter() {
            println!("{:?}: {}", config, x_c);
        }
    }

    println!("Merging configurations");
    let wide = GeneralizedSelection::merge(stacks);
    let narrow = NarrowJobSelection::merge(narrow_jobs);
    println!("Done merging configurations");
    println!(
        "Done reducing resource amounts resulting in {} entries in the wide job selection and {} entries in the narrow job selection",
         wide.configurations.len(), narrow.processing_times.len());
    (wide, narrow)
}

fn group_by_machine_count(
    problem: &ProblemData,
    x_tilde: &GeneralizedSelection,
) -> (f64, Vec<(f64, GeneralizedSelection)>) {
    println!("Grouping by machine count");
    let m = problem.machine_count_usize;
    println!("m={} and 1/e'={}", m, problem.one_over_epsilon_prime);
    let k_len = min(m, problem.one_over_epsilon_prime as usize);
    println!("Creating {k_len} sets");
    // List of K_i sets with pre-computed P_pre(K_i) per set, where i+1 than the number of machines
    let mut k: Vec<(f64, GeneralizedSelection)> = vec![(0.0, GeneralizedSelection::empty()); k_len];
    let mut p_pre = 0.0;
    for (c, x_c) in x_tilde
        .configurations
        .iter()
        .filter(|(c, _)| c.configuration.machine_count > 0)
    {
        let i = c.configuration.machine_count as usize;
        println!(
            "{i} machines used in config {:?} which was selected {x_c}",
            c
        );
        let group = i - 1;
        p_pre += x_c;
        k[group].0 += x_c;
        k[group].1.push(c.clone(), *x_c);
    }
    println!("Done creating sets with p_pre={p_pre}");

    println!("Sorting groups");
    for (_, k_i) in k.iter_mut() {
        k_i.sort_by_resource_amount();
    }
    println!("Done sorting groups");

    println!("Done grouping by machine count");
    (p_pre, k)
}

struct Grouping {
    group_size: f64,
    groups: Vec<(usize, Vec<Job>)>,
}
impl Grouping {
    fn new(problem: &ProblemData, groups: Vec<Vec<Job>>) -> Self {
        Grouping {
            group_size: problem.one_over_epsilon_prime as f64,
            groups: groups.into_iter().map(|group| (0, group)).collect(),
        }
    }
    fn next(&mut self, job: Job) -> Option<Job> {
        let i = (job.resource_amount % self.group_size) as usize;
        if self.groups.len() <= i {
            return None;
        }
        if self.groups[i].0 == self.groups[i].1.len() {
            return None;
        }
        let res = self.groups[i].1[self.groups[i].0];
        self.groups[i].0 += 1;
        Some(res)
    }
}

fn group_by_resource_amount(problem: &ProblemData) -> Grouping {
    let ProblemData {
        ref jobs,
        epsilon_prime_squared,
        ..
    } = *problem;

    let group_size = epsilon_prime_squared as f64;
    println!(
        "Grouping by resource amount into groups of size {}",
        group_size
    );
    let len = 1.0 / group_size; // G
    let mut groups: Vec<Vec<Job>> = vec![vec![]; len as usize];
    for job in jobs.iter().copied() {
        let r = job.resource_amount;
        let i = r % group_size;
        groups[i as usize].push(job)
    }

    let g = Grouping::new(problem, groups);
    println!("Created {} groups", g.groups.len());
    g
}

fn integral_schedule(
    problem: &ProblemData,
    x_bar: GeneralizedSelection,
    y_bar: NarrowJobSelection,
) -> Schedule {
    println!("Computing integral schedule");
    let mut s = Schedule::empty(problem.machine_count_usize);

    let x_hat = GeneralizedSelection {
        configurations: x_bar
            .configurations
            .into_iter()
            .map(|(c, x_c)| (c, x_c + problem.p_max))
            .collect(),
    };

    println!(
        "Grouping {} entries in x_hat by windows",
        x_hat.configurations.len()
    );
    let window_groups = x_hat
        .configurations
        .into_iter()
        .fold(HashMap::new(), |mut agg, sel| {
            let key = sel.0.window;
            let val = (sel.0.configuration, sel.1);
            match agg.get_mut(&key) {
                None => {
                    agg.insert(key, vec![val]);
                }
                Some(ls) => {
                    ls.push(val);
                }
            }
            agg
        });
    println!("Obtained {} window groups", window_groups.len());

    let mut groups = group_by_resource_amount(problem);
    println!("Finding jobs");
    for (win, configs) in window_groups {
        let mut p_w = problem.p_max;
        println!("Looking at window {:?} with p_w={}", win, p_w);
        println!("Adding wide jobs");
        for (c, x_c) in configs {
            println!("  Adding config {:?} with x_c={}", c, x_c);
            p_w += x_c;
            let m = c.machine_count as usize;
            let off = 0;
            let utilization = vec![0.0; m];
            for (machine, job) in c
                .jobs
                .into_iter()
                .flat_map(|(job, i)| repeat(job).take(i as usize))
                .enumerate()
            {
                print!("    Finding location for {:?} ...", job);
                if let Some(job) = groups.next(job) {
                    let found = 'search: {
                        for i in 0..machine {
                            let target = (i + off) % m;
                            if utilization[target] <= x_c {
                                println!(" found at machine {target}.");
                                s.add(target, job);
                                break 'search true;
                            }
                        }
                        false
                    };
                    if !found {
                        println!(" not found, falling back to {machine}");
                        s.add(machine, job);
                    }
                } else {
                    println!(" no matching job found in grouping!");
                }
            }
        }

        println!("Adding narrow jobs");
        let mut narrow_jobs = y_bar.get_jobs_by_window(&win);
        narrow_jobs.sort_by(|(job0, _), (job1, _)| {
            job0.resource_amount
                .partial_cmp(&job1.resource_amount)
                .expect("bad resource amount, cannot sort")
                .reverse() // sort by decreasing resource amount
        });
        let mut processing_time = 0.0;
        let mut target_machine = problem.machine_count_usize - 1;
        println!("  Starting at machine {target_machine}, limited at processing time p_w={p_w}");
        for (job, p) in narrow_jobs {
            if processing_time + p > p_w {
                println!("  Machine {target_machine} full because processing time is {processing_time}, stepping back");
                processing_time = 0.0;
                target_machine -= 1;
            }
            println!("  Adding {:?} to {target_machine}", job);
            s.add(target_machine, job);
            processing_time += p;
        }
    }

    println!("Done creating integral schedule");
    s
}
