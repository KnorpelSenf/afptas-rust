use console::style;
use good_lp::{
    constraint, default_solver, variable, variables, Expression, ProblemVariables, Solution,
    SolverModel, Variable,
};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use log::{
    debug, log_enabled, trace,
    Level::{Info, Trace},
};
use std::{
    cmp::{max, min, Ordering},
    collections::{HashMap, HashSet},
    fmt::{Debug, Write},
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
    pub jobs: Vec<InstanceJob>,
}
#[derive(Debug, Copy, Clone)]
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
    fn is_wide(&self, job: &Job) -> bool {
        self.second_case || job.resource_amount >= self.epsilon_prime_times_resource_limit
    }
}
impl From<Instance> for ProblemData {
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
    pub machine_count: usize,
    pub chunks: Vec<ScheduleChunk>,
}
impl Schedule {
    fn empty(problem: &ProblemData) -> Self {
        Schedule {
            machine_count: problem.machine_count_usize,
            chunks: vec![],
        }
    }
    fn push(&mut self, chunk: ScheduleChunk) {
        self.chunks.push(chunk);
    }
    fn push_all(&mut self, chunks: Vec<ScheduleChunk>) {
        self.chunks.extend(chunks);
    }

    fn make_chunk(&self) -> ScheduleChunk {
        ScheduleChunk {
            machines: vec![MachineSchedule::empty(); self.machine_count],
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScheduleChunk {
    pub machines: Vec<MachineSchedule>,
}
impl ScheduleChunk {
    fn add(&mut self, machine: usize, job: Job) {
        self.machines[machine].push(job)
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
    debug!(
        "Computing schedule from {} wide and {} narrow jobs:",
        wide_jobs.len(),
        narrow_jobs.len()
    );
    trace!("Wide jobs are: {:?}", wide_jobs);
    trace!("Narrow jobs are: {:?}", narrow_jobs);

    let job_len = problem_data.jobs.len();
    let x = max_min(&problem_data);
    trace!("Max-min solved with:");
    print_selection(job_len, problem_data.machine_count_usize, &x);
    let x = reduce_to_basic_solution(x);
    trace!("Reduced the max-min solution min to:");
    print_selection(job_len, problem_data.machine_count_usize, &x);
    let (x_tilde, y_tilde) = generalize(&problem_data, x);
    trace!("Generalized to:");
    print_gen_selection(
        job_len,
        problem_data.machine_count_usize,
        problem_data.resource_limit,
        &x_tilde,
    );
    let (x_bar, y_bar) = reduce_resource_amounts(&problem_data, &x_tilde, &y_tilde);
    // TODO: use x_bar, y_bar to find a basic solution to LP_w via simplex
    integral_schedule(&problem_data, x_bar, y_bar)
}

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
        // ((1-tau) * solution) + (tau * solution)
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
    debug!("Solving max-min");
    let ProblemData {
        epsilon,
        epsilon_squared,
        epsilon_prime,
        one_over_epsilon_prime,
        ref jobs,
        ..
    } = *problem_data;

    let progress_bar = !log_enabled!(Info);
    if progress_bar {
        println!(
            "{} Finding initial solution ...",
            style("[1/5]").bold().dim()
        );
    }
    // compute initial solution;
    debug!("Computing initial solution");
    let m = jobs.len();
    let pb = if progress_bar {
        ProgressBar::new(m as u64)
    } else {
        ProgressBar::hidden()
    };
    let mut x = Selection::init((0..m).map(|i| {
        pb.inc(1);
        (
            solve_block_problem_ilp(&unit(i, m), &problem_data),
            1.0 / m as f64,
        )
    }));
    pb.finish_and_clear();
    debug!("Done computing initial solution");
    trace!("Initial value is {x:?}");

    if progress_bar {
        println!("{} Improving solution ...", style("[2/5]").bold().dim());
    }

    let pb = if progress_bar {
        ProgressBar::new(one_over_epsilon_prime as u64 * 10000).with_style(
            ProgressStyle::with_template(&format!(
                "[{{elapsed_precise}}] {{spinner}} Approaching {} ... {{msg}}",
                epsilon_prime
            ))
            .unwrap()
            .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
            }),
        )
    } else {
        ProgressBar::hidden()
    };

    debug!("Iterating until v < {epsilon_prime} = epsilon_prime");
    // iterate
    loop {
        let fx: Vec<f64> = jobs.iter().map(|job| f(job, &x)).collect();
        trace!("f(x) = {fx:?}");
        // price vector
        let prec = epsilon_squared / (m as f64);
        let theta = find_theta(epsilon_prime, &fx, prec);
        let price = compute_price(&fx, epsilon_prime, theta);
        trace!("++ Starting iteration with price {price:?}");
        // solve block problem
        let config = solve_block_problem_ilp(&price, &problem_data);
        let y = Selection::single(config);
        trace!("Received block problem solution {y:?}");
        let fy: Vec<f64> = jobs.iter().map(|job| f(job, &y)).collect();
        trace!("f(y) = {fy:?}");

        // compute v
        let v = compute_v(&price, &fx, &fy);
        pb.set_position((10000.0 / v) as u64);
        pb.set_message(v.to_string());
        debug!("v = {v}");
        if v < epsilon_prime {
            let λ_hat = fx
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .unwrap_or(&1.0);
            debug!("λ^ = {}", λ_hat);
            pb.finish_and_clear();
            x.scale(1.0 / λ_hat);
            break;
        }
        // update solution = ((1-tau) * solution) + (tau * solution)
        let tau = line_search(&fx, &fy, theta, epsilon_prime, epsilon);
        x.interpolate(y, tau);
        trace!(
            "Updated solution with step length tau={} to be {:?}",
            tau,
            x
        );
    }
    x
}
fn print_selection(job_len: usize, m: usize, x: &Selection) {
    if log_enabled!(Trace) {
        let digits_per_job_id = (job_len - 1).to_string().len();
        let lcol = max(4, digits_per_job_id * m);
        trace!("{: >lcol$} | Length", "Jobs");
        trace!("{:->lcol$}---{}", "-", "-".repeat(19));
        for (c, x_c) in x.0.iter() {
            let job_ids = c
                .jobs
                .iter()
                .map(|job| {
                    format!("{: >digits_per_job_id$}", job.0.id.to_string()).repeat(job.1 as usize)
                })
                .collect::<Vec<_>>()
                .join("");
            trace!("{: >lcol$} | {}", job_ids, x_c);
        }
    }
}
fn print_gen_selection(job_len: usize, m: usize, r: f64, x: &GeneralizedSelection) {
    if log_enabled!(Trace) {
        let digits_per_job_id = (job_len - 1).to_string().len();
        let digits_per_machine = m.to_string().len();
        let resource_precision = 2;
        let digits_per_resource = r.ceil().to_string().len() + 1 + resource_precision;
        let lcol = max("Jobs".len(), digits_per_job_id * m);
        let mcol = max(
            "w=(m,r)".len(),
            1 + digits_per_machine + ", ".len() + digits_per_resource + 1,
        );
        trace!("{:>lcol$} | {:<mcol$} | Length", "Jobs", "w=(m,r)",);
        trace!("{:->lcol$}---{:-<mcol$}---{}", "-", "-", "-".repeat(19));
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
            trace!("{:>lcol$} | {win} | {}", job_ids, x_c);
        }
    }
}

fn solve_block_problem_ilp(q: &Vec<f64>, problem_data: &ProblemData) -> Configuration {
    let ProblemData {
        machine_count,
        resource_limit,
        one_over_epsilon_prime,
        ref jobs,
        ..
    } = *problem_data;
    let wide_job_max_count = one_over_epsilon_prime;
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
    trace!("Solved ILP for {} items with {:?}", q.len(), a_star);
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
    debug!("Reducing to basic solution");
    // TODO: put the output of this ILP_kkp into a simplex implementation in order to obtain a basic solution
    debug!("Done reducing to basic solution");
    x
}

fn generalize(problem: &ProblemData, x: Selection) -> (GeneralizedSelection, NarrowJobSelection) {
    let narrow_jobs: Vec<Job> = problem
        .jobs
        .iter()
        .filter(|job| !problem.is_wide(job))
        .copied()
        .collect();
    let progress_bar = !log_enabled!(Info);
    if progress_bar {
        println!("{} Generalizing ...", style("[3/5]").bold().dim());
    }
    let pb = if progress_bar {
        ProgressBar::new(x.0.len() as u64)
    } else {
        ProgressBar::hidden()
    };
    let (x_tilde, y_tilde) =
        x.0.iter()
            .map(|(c, x_c)| {
                pb.inc(1);
                (c, c.reduce_to_wide_jobs(problem), x_c)
            })
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
                    *acc_x.entry(gen).or_insert(0.0) += *x_c;

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
    fn get_jobs_by_window(&self, win: &Window) -> Vec<(Job, f64)> {
        let window = *win;
        if let Some(jobs) = self.windex.get(win) {
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
    fn find_max_occurrences(self) -> HashMap<Window, Vec<(Job, f64)>> {
        let mut max = HashMap::new();
        for (config, p) in self.processing_times {
            let current = max.entry(config.narrow_job).or_insert((config.window, p));
            if current.1 < p {
                current.0 = config.window;
                current.1 = p;
            }
        }
        max.into_iter()
            .fold(HashMap::new(), |mut agg, (job, (window, p))| {
                agg.entry(window).or_insert(vec![]).push((job, p));
                agg
            })
    }
    fn add(&mut self, config: NarrowJobConfiguration, processing_time: f64) {
        *self.processing_times.entry(config).or_insert(0.0) += processing_time;
        self.windex
            .entry(config.window)
            .or_insert(HashSet::new())
            .insert(config.narrow_job);
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
    debug!("Reducing resource amounts");
    let (p_pre, k) = group_by_machine_count(problem, x_tilde);

    debug!("p_pre={p_pre}");
    let step_width = problem.epsilon_prime_squared * p_pre;
    debug!("Step width is {step_width}");

    let progress_bar = !log_enabled!(Info);
    if progress_bar {
        println!(
            "{} Reducing resource amounts ...",
            style("[4/5]").bold().dim()
        );
    }
    let pb = if progress_bar {
        ProgressBar::new(k.len() as u64)
    } else {
        ProgressBar::hidden()
    };

    let (stacks, narrow_jobs): (Vec<GeneralizedSelection>, Vec<NarrowJobSelection>) = k
        .into_iter()
        .map(|(x_c, k_i)| {
            pb.inc(1);
            trace!("  Processing generalized selection with x_c={x_c}");
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
                        trace!("    Processing configuration {:?} with p={p}", c);
                        let s_c = e_c - p;
                        let cur_step = (e_c / step_width).floor();
                        let next_step = (s_c / step_width).floor();
                        let is_cut = cur_step != next_step;

                        trace!("    s_c={s_c} --- {e_c}=e_c");
                        if is_cut {
                            trace!("    current={cur_step} --- {next_step}=next ++ CUT! ++");
                        } else {
                            trace!("    current={cur_step} --- {next_step}=next");
                        }

                        k_i.push(c.clone());

                        if is_cut {
                            let lowest_cut = (s_c / step_width).ceil() * step_width;
                            trace!("    lowest cut={lowest_cut}");
                            let p_w_ik: f64 = k_i
                                .iter()
                                .map(|k_ik| {
                                    trace!(
                                        "    Adding narrow jobs from K_ik by window {:?}",
                                        k_ik.window
                                    );
                                    for (narrow_job, amount) in
                                        y_tilde.get_jobs_by_window(&k_ik.window).iter()
                                    {
                                        trace!(
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
                                    }

                                    k_ik.configuration.processing_time
                                })
                                .sum();
                            k_i = vec![];

                            let w_down = e_c - lowest_cut;
                            let phi_down = w_down / p_w_ik;
                            let phi_up = 1.0 - phi_down;
                            let next_window = c.window;
                            trace!("    Next window will be {:?}", c.window);

                            trace!("    Adding narrow jobs by window {:?}", window);
                            for (narrow_job, amount) in y_tilde.get_jobs_by_window(&window).iter() {
                                trace!(
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
                                trace!(
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
                            }

                            let highest_cut = cur_step * step_width;
                            let p_diff = e_c - highest_cut;
                            trace!("    Adding generalized config with x_c={p_diff}");
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

                        trace!("    Adding generalized config with x_c={p}");
                        wide_sel.push(
                            GeneralizedConfiguration {
                                configuration: c.configuration,
                                window,
                            },
                            p,
                        );

                        trace!("    Done processing configuration, p is now {p}");
                        (wide_sel, narrow_sel, s_c, window, k_i)
                    },
                );
            trace!(
                "  Obtained a selection with {} wide jobs and a selection with {} narrow jobs",
                sel.configurations.len(),
                narrow.processing_times.len()
            );
            trace!("  Done processing generalized selection with x_c={x_c}");
            (sel, narrow)
        })
        .unzip();
    pb.finish_and_clear();

    debug!("Done creating {} stacks", stacks.len());
    for (i, stack) in stacks.iter().enumerate() {
        trace!("  --- K_{i} ---  ");
        for (config, x_c) in stack.configurations.iter() {
            trace!("{:?}: {}", config, x_c);
        }
    }

    debug!("Merging configurations");
    let wide = GeneralizedSelection::merge(stacks);
    let narrow = NarrowJobSelection::merge(narrow_jobs);
    debug!("Done merging configurations");
    debug!(
        "Done reducing resource amounts resulting in {} entries in the wide job selection and {} entries in the narrow job selection",
         wide.configurations.len(), narrow.processing_times.len());
    (wide, narrow)
}

fn group_by_machine_count(
    problem: &ProblemData,
    x_tilde: &GeneralizedSelection,
) -> (f64, Vec<(f64, GeneralizedSelection)>) {
    debug!("Grouping by machine count");
    let m: usize = problem.machine_count_usize;
    debug!("m={} and 1/e'={}", m, problem.one_over_epsilon_prime);
    let k_len = min(m, problem.one_over_epsilon_prime as usize);
    debug!("Creating {k_len} sets");
    // List of K_i sets with pre-computed P_pre(K_i) per set, where i than the number of machines
    let mut k: Vec<(f64, GeneralizedSelection)> = vec![(0.0, GeneralizedSelection::empty()); k_len];
    // TODO: this also has to happen for the last window (R,m), which should be
    // x_bar(emptyset, (R,m)) = x_tilde(emptyset, (R,m)) + epsilon_prime * P_pre
    let mut p_pre = 0.0;
    for (c, x_c) in x_tilde
        .configurations
        .iter()
        .filter(|(c, _)| c.configuration.machine_count > 0)
    {
        let group = c.configuration.machine_count as usize - 1;
        trace!(
            "{group} machines used in config {:?} which was selected {x_c}",
            c
        );
        p_pre += x_c;
        k[group].0 += x_c;
        k[group].1.push(c.clone(), *x_c);
    }
    debug!("Done creating sets with p_pre={p_pre}");

    debug!("Sorting groups");
    for (_, k_i) in k.iter_mut() {
        k_i.sort_by_resource_amount();
    }
    debug!("Done sorting groups");

    debug!("Done grouping by machine count");
    (p_pre, k)
}

struct Grouping {
    groups: Vec<(usize, Vec<Job>)>,
    map: HashMap<Job, usize>,
}
impl Grouping {
    fn new(groups: Vec<Vec<Job>>) -> Self {
        let mut map = HashMap::new();
        for (i, group) in groups.iter().enumerate() {
            for job in group {
                map.insert(job.clone(), i);
            }
        }

        Grouping {
            groups: groups.into_iter().map(|group| (0, group)).collect(),
            map,
        }
    }
    fn next(&mut self, job: Job) -> Option<Job> {
        let i = *self.map.get(&job)?;
        let group = self.groups.get(i)?;
        let res = *group.1.get(group.0)?;
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
    debug!(
        "Grouping by resource amount into groups of size {}",
        group_size
    );
    let len = 1.0 / group_size; // G
    let mut groups: Vec<Vec<Job>> = vec![vec![]; len as usize];
    let mut wide_jobs: Vec<Job> = jobs
        .iter()
        .filter(|job| problem.is_wide(job))
        .copied()
        .collect();
    wide_jobs.sort_by(|job0, job1| {
        job0.resource_amount
            .partial_cmp(&job1.resource_amount)
            .expect("bad resource amount, cannot sort")
    });
    let p_w: f64 = wide_jobs.iter().map(|job| job.processing_time).sum();
    let mut current_p = 0.0;
    let mut current_i: usize = 0;

    for job in wide_jobs {
        groups[current_i].push(job);
        current_p += job.processing_time;
        if current_p >= ((current_i + 1) as f64) * group_size * p_w {
            current_i = min(groups.len() - 1, (current_p / (group_size * p_w)) as usize);
        }
    }

    let g = Grouping::new(groups);
    debug!("Created {} groups", g.groups.len());
    g
}

fn integral_schedule(
    problem: &ProblemData,
    x_bar: GeneralizedSelection,
    y_bar: NarrowJobSelection,
) -> Schedule {
    debug!("Computing integral schedule");
    let ProblemData {
        p_max,
        machine_count_usize,
        ..
    } = *problem;
    let mut s = Schedule::empty(problem);

    let mut full_chunk = s.make_chunk();

    if !log_enabled!(Info) {
        println!(
            "{} Computing integral schedule ...",
            style("[5/5]").bold().dim()
        );
    }

    // Add p_max everywhere
    let x_hat = GeneralizedSelection {
        configurations: x_bar
            .configurations
            .into_iter()
            .map(|(c, x_c)| (c, x_c + p_max))
            .collect(),
    };

    // Group wide jobs by windows
    debug!(
        "Grouping {} entries in x_hat by windows",
        x_hat.configurations.len()
    );
    let window_groups: HashMap<Window, Vec<(Configuration, f64)>> = x_hat
        .configurations
        .into_iter()
        .fold(HashMap::new(), |mut agg, sel| {
            agg.entry(sel.0.window)
                .or_insert(vec![])
                .push((sel.0.configuration, sel.1));
            agg
        });
    debug!("Obtained {} window groups", window_groups.len());

    let narrow_jobs_by_window = y_bar.find_max_occurrences();

    // Put wide jobs into chunks
    let mut groups = group_by_resource_amount(problem);
    trace!("{:?}", groups.groups);
    debug!("Finding jobs");
    for (win, configs) in window_groups {
        debug!("+++ Looking at window {:?}", win);
        debug!("Creating chunks with wide jobs");
        let mut chunks: Vec<(ScheduleChunk, f64)> = configs
            .into_iter()
            .map(|(c, x_c)| {
                let mut chunk = s.make_chunk();
                trace!("  Adding config {:?} with x_c={}", c, x_c);
                for (machine, job) in c
                    .jobs
                    .into_iter()
                    .flat_map(|(job, i)| repeat(job).take(i as usize))
                    .enumerate()
                {
                    if let Some(found_job) = groups.next(job) {
                        trace!(
                            "    Found location {} for job {:?}, adding it to machine {machine}",
                            found_job.id,
                            job
                        );
                        chunk.add(machine, found_job);
                    } else {
                        trace!("    No matching job found in grouping for job {:?}!", job);
                    }
                }
                (chunk, x_c)
            })
            .collect();
        debug!("Done creating {} chunks with wide jobs", chunks.len());

        // Put narrow jobs into windows
        debug!("Adding narrow jobs");
        let mut narrow_jobs = narrow_jobs_by_window.get(&win).unwrap_or(&vec![]).clone();
        let narrow_job_count = narrow_jobs.len();
        narrow_jobs.sort_by(|(job0, _), (job1, _)| {
            job0.resource_amount
                .partial_cmp(&job1.resource_amount)
                .expect("bad resource amount, cannot sort")
                .reverse() // sort by decreasing resource amount
        });
        debug!(
            "Found jobs {:?} for window {:?}",
            narrow_jobs.iter().map(|pair| pair.0).collect::<Vec<_>>(),
            win
        );
        let mut used_processing_time = 0.0;
        let mut target_chunk = 0;
        let mut target_machine = machine_count_usize - 1;
        for (job, p) in narrow_jobs {
            trace!(
                "  Looking at {:?} with processing time {} which is scheduled for {}",
                job,
                job.processing_time,
                p
            );
            if job.processing_time < p {
                trace!("  Adding {:?} to 0 (full chunk)", job);
                full_chunk.add(0, job)
            } else {
                if used_processing_time + job.processing_time > chunks[target_chunk].1 {
                    used_processing_time = 0.0;
                    if target_chunk == chunks.len() - 1 {
                        trace!("  Machine {target_machine} full, stepping back");
                        target_chunk = 0;
                        target_machine -= 1;
                    } else {
                        target_chunk += 1;
                    }
                }
                trace!("  Adding {:?} to {target_machine}", job);
                chunks[target_chunk].0.add(target_machine, job);
                used_processing_time += job.processing_time;
            }
        }
        debug!("Done adding {narrow_job_count} narrow jobs");
        s.push_all(chunks.into_iter().map(|(chunk, _)| chunk).collect());
        debug!("Done looking at window {:?}", win);
    }

    s.push(full_chunk);

    debug!("Done creating integral schedule");
    s
}
