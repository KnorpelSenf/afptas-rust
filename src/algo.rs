use good_lp::{
    constraint, default_solver, variable, variables, Expression, ProblemVariables, Solution,
    SolverModel, Variable,
};
use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::Debug,
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

#[derive(Clone)]
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

    let _i_sup = create_i_sup(
        problem_data
            .jobs
            .iter()
            .filter(|job| problem_data.is_wide(job))
            .map(|job| Rc::clone(job))
            .collect(),
        &problem_data,
    );

    let x = max_min(&problem_data);
    let x_tilde = generalize(&problem_data, x);
    println!("{:?}", x_tilde);

    Schedule {
        mapping: Box::from(vec![]),
    }
}

fn create_i_sup(wide_jobs: Vec<Rc<Job>>, problem_data: &ProblemData) -> Vec<Rc<Job>> {
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
            Rc::from(Job {
                id: job_ids.next().unwrap(),
                processing_time: step,
                resource_amount,
            })
        })
        .collect::<Vec<_>>();
    println!(
        "Obtained I_sup with {} additional jobs, totalling {} jobs",
        additional_jobs.len(),
        wide_jobs.len() + additional_jobs.len(),
    );
    [wide_jobs, additional_jobs].concat()
}

fn linear_grouping(step: f64, jobs: &Vec<Rc<Job>>) -> Vec<Vec<Rc<Job>>> {
    let n = jobs.len();
    println!("Grouping {} jobs: {jobs:?}", n);
    // FIXME: Add special handling for the last group since we already know that
    // all remaining jobs will be put into it. Due to floating point
    // imprecision, it might happen that we accidentally open one group to many,
    // containing a single job only, having the size of the floating point
    // rounding error.
    if n == 0 {
        return vec![];
    }
    let mut job_ids = 0..;

    let mut groups: Vec<Vec<Rc<Job>>> = vec![];
    let mut current_group: Vec<Rc<Job>> = vec![];
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
                current_group.push(Rc::from(new_job));
                break;
            }

            // Split off a bit of the job for the current group
            let new_job = Job {
                id: job_ids.next().unwrap(),
                processing_time: remaining_space,
                resource_amount: job.resource_amount,
            };
            current_group.push(Rc::from(new_job));
            groups.push(current_group);

            current_group = vec![];

            current_processing_time += remaining_space;
            remaining_processing_time -= remaining_space;
        }
    }
    println!("Obtained {} groups", groups.len());
    groups
}

#[derive(Clone)]
struct Configuration {
    /// job -> index in vector
    index: HashMap<Rc<Job>, usize>,
    /// job -> how many times it is contained
    jobs: Vec<(Rc<Job>, i32)>,
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
    fn new(jobs: Vec<(Rc<Job>, i32)>) -> Self {
        if jobs.windows(2).any(|pair| pair[0].0.id >= pair[1].0.id) {
            panic!("jobs are out of order");
        }
        let (processing_time, resource_amount, machine_count) =
            jobs.iter().fold((0.0, 0.0, 0), |(p, r, m), (job, count)| {
                (
                    p + job.processing_time * *count as f64,
                    r + job.resource_amount * *count as f64,
                    m + *count,
                )
            });
        let index = HashMap::from_iter(
            jobs.iter()
                .enumerate()
                .map(|(i, (job, _))| (Rc::clone(job), i)),
        );
        Configuration {
            jobs,
            index,
            processing_time,
            resource_amount,
            machine_count,
        }
    }
    /// C(j)
    fn job_count(&self, job: &Rc<Job>) -> i32 {
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

fn f(j: &Rc<Job>, x: &Selection) -> f64 {
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
    println!("Max-min solved with {:?}", x);
    x
}

fn solve_block_problem_ilp(
    q: &Vec<f64>,
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
        .zip(q.iter())
        .filter(|(_, &q)| q > 0.0)
        .map(|(job, &q)| {
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
        .filter(|(_, var)| *var != 0)
        .collect();
    println!("Solved ILP for {:?} with {:?}", q, a_star);
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

fn generalize(problem: &ProblemData, x: Selection) -> GeneralizedSelection {
    GeneralizedSelection::from_iter(
        x.0.iter()
            .map(|(c, x_c)| (c.reduce_to_wide_jobs(problem), x_c))
            .fold(HashMap::new(), |mut acc, (c, x_c)| {
                let gen = GeneralizedConfiguration::from_configuration(problem, c);
                let existing = acc.get(&gen).unwrap_or(&0.0);
                acc.insert(gen, x_c + existing);
                acc
            }),
    )
}

struct Window {
    resource_amount: f64,
    machine_count: i32,
}
impl Window {
    fn main(problem: &ProblemData, config: &Configuration) -> Self {
        Window {
            resource_amount: problem.resource_limit - config.resource_amount,
            machine_count: problem.machine_count - config.machine_count,
        }
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

struct GeneralizedConfiguration {
    configuration: Configuration,
    winodw: Window,
}
impl GeneralizedConfiguration {
    fn from_configuration(problem: &ProblemData, configuration: Configuration) -> Self {
        let main_window = Window::main(problem, &configuration);
        GeneralizedConfiguration {
            configuration,
            winodw: main_window,
        }
    }
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
        f.write_str("GConfig")?;
        self.winodw.fmt(f)?;
        f.debug_list().entries(&self.configuration.jobs).finish()?;
        Ok(())
    }
}
#[derive(Debug)]
struct GeneralizedSelection(HashMap<GeneralizedConfiguration, f64>);
impl GeneralizedSelection {
    fn from_iter<T: IntoIterator<Item = (GeneralizedConfiguration, f64)>>(iter: T) -> Self {
        GeneralizedSelection(HashMap::from_iter(iter))
    }
}
