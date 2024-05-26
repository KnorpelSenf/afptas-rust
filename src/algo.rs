use good_lp::{
    constraint, default_solver, variable, variables, Expression, ProblemVariables, Solution,
    SolverModel, Variable,
};
use std::{
    cmp::{max, min, Ordering},
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
        resource_limit,
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

    let job_len = problem_data.jobs.len();
    let x = max_min(&problem_data);
    println!("Max-min solved with:");
    print_selection(job_len, machine_count, &x);
    let (x_tilde, y_tilde) = generalize(&problem_data, x);
    println!("Generalized to:");
    print_gen_selection(job_len, machine_count, resource_limit, &x_tilde);
    // for x in x_tilde.0.iter() {
    //     println!("{:?}", x);
    // }
    let _x_bar = reduce_resource_amounts(&problem_data, &x_tilde);
    let _y_bar = assign_narrow_jobs(&x_tilde, &y_tilde);

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
    fn empty() -> Self {
        Configuration {
            jobs: vec![],
            index: HashMap::new(),
            processing_time: 0.0,
            resource_amount: 0.0,
            machine_count: 0,
        }
    }
    fn new(jobs: Vec<(Rc<Job>, i32)>) -> Self {
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
    x
}
fn print_selection(job_len: usize, m: i32, x: &Selection) {
    let digits_per_job_id = (job_len - 1).to_string().len();
    let lcol = max(4, digits_per_job_id * m as usize);
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
fn print_gen_selection(job_len: usize, m: i32, r: f64, x: &GeneralizedSelection) {
    let digits_per_job_id = (job_len - 1).to_string().len();
    let digits_per_machine = m.to_string().len();
    let resource_precision = 2;
    let digits_per_resource = r.ceil().to_string().len() + 1 + resource_precision;
    let lcol = max("Jobs".len(), digits_per_job_id * m as usize);
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

fn generalize(problem: &ProblemData, x: Selection) -> (GeneralizedSelection, NarrowJobSelection) {
    let narrow_jobs: Vec<Rc<Job>> = problem
        .jobs
        .iter()
        .filter(|job| !problem.is_wide(job))
        .map(|job| Rc::clone(job))
        .collect();
    let (x_tilde, y_tilde) =
        x.0.iter()
            .map(|(c, x_c)| (c, c.reduce_to_wide_jobs(problem), x_c))
            .fold(
                (HashMap::new(), HashMap::new()),
                |(mut acc_x, mut acc_y), (c, c_w, x_c)| {
                    let win = Rc::new(Window::main(problem, &c_w));

                    for narrow_job in narrow_jobs.iter() {
                        let nconf = NarrowJobConfiguration {
                            narrow_job: Rc::clone(&narrow_job),
                            window: Rc::clone(&win),
                        };
                        let existing = acc_y.get(&nconf).unwrap_or(&0.0);
                        acc_y.insert(nconf, c.job_count(&narrow_job) as f64 * x_c + existing);
                    }

                    let gen = GeneralizedConfiguration {
                        configuration: c_w,
                        window: win,
                    };
                    let existing = acc_x.get(&gen).unwrap_or(&0.0);
                    acc_x.insert(gen, x_c + existing);

                    (acc_x, acc_y)
                },
            );

    (
        GeneralizedSelection::from(x_tilde),
        NarrowJobSelection(y_tilde),
    )
}

#[derive(Clone)]
struct Window {
    resource_amount: f64,
    machine_count: i32,
}
impl Window {
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
    window: Rc<Window>,
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
    index: HashMap<GeneralizedConfiguration, usize>,
}
impl GeneralizedSelection {
    fn new() -> Self {
        GeneralizedSelection {
            configurations: vec![],
            index: HashMap::new(),
        }
    }
    fn from(mapping: HashMap<GeneralizedConfiguration, f64>) -> Self {
        let mut sel = GeneralizedSelection::new();
        for (config, x_c) in mapping.into_iter() {
            sel.set(config, x_c);
        }
        sel
    }
    fn get(&self, config: &GeneralizedConfiguration) -> Option<f64> {
        Some(self.configurations[*self.index.get(config)?].1)
    }
    fn set(&mut self, config: GeneralizedConfiguration, x_c: f64) {
        let posiiton = self.index.entry(config).or_insert_with(|| {
            let pos = self.configurations.len();
            self.configurations.push((config, 0.0));
            pos
        });
        self.configurations[*posiiton].1 = x_c;
    }
}

#[derive(PartialEq, Eq, Hash)]
struct NarrowJobConfiguration {
    narrow_job: Rc<Job>,
    window: Rc<Window>,
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
struct NarrowJobSelection(HashMap<NarrowJobConfiguration, f64>);

fn reduce_resource_amounts(
    problem: &ProblemData,
    x_tilde: &GeneralizedSelection,
) -> Vec<Vec<GeneralizedSelection>> {
    // FIXME: the above return type is bullshit, we have to return a
    // GeneralizedSelection instead of a GeneralizedConfiguration. We currently
    // throw away the runtime for each gen config, which we should not do. Hint:
    // we may want to refactor GeneralizedSelection to store an order with its
    // elements so that we no longer have to invent a different intermediate
    // data strucutre in this procedure. Doing that would allow us to throw away
    // the vector of pairs.
    println!("Reducing resource amounts");
    let m = problem.machine_count as usize;
    println!("m={} and 1/e'={}", m, problem.one_over_epsilon_prime);
    let k_len = min(m, problem.one_over_epsilon_prime as usize);
    // List of K_i sets with pre-computed P_pre(K_i) per set, where i+1 than the number of machines,
    // and using a vector of pairs as an ordered version of a generalized selection.
    let mut k: Vec<(f64, Vec<(&GeneralizedConfiguration, &f64)>)> = vec![(0.0, vec![]); k_len];
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
        k[group].1.push((&c, x_c));
    }

    for (_, k_i) in k.iter_mut() {
        k_i.sort_by(|c0, c1| {
            c0.0.configuration
                .resource_amount
                .partial_cmp(&c1.0.configuration.resource_amount)
                .expect(&format!(
                    "could not comare resource amounts {} and {}",
                    c0.0.configuration.resource_amount, c1.0.configuration.resource_amount
                ))
        });
    }

    let p_pre = p_pre; // end mut
    let k = k; // end mut
    println!("P_pre={p_pre}");
    let step_width = problem.epsilon_prime_squared * p_pre;
    println!("Step width is {step_width}");

    let stacks: Vec<GeneralizedConfiguration> =
        k.into_iter().fold(vec![], |mut stacks, (sum, configs)| {
            // we fold the stack top-to-bottom, reducing the processing time at every step,
            // and reducing the resouce amount as well as k whenever we make a cut
            let (mut stack, _, _, _) = configs.into_iter().rev().fold(
                (vec![], sum, 0.0, (sum / step_width).ceil() - 1.0),
                // p: the current processing time
                // r: resource amount at last cut
                |(mut stack, p, r, k), c| {
                    let mut cut = k * step_width;
                    let processing_time = c.1;
                    let end = p - processing_time;
                    let (r, cuts_done) = if end < cut {
                        let mut cuts_done = 0.0;
                        let mut last_cut = p;
                        // FIXME: this is bullshit, we do not have to cut at
                        // every intersection. Instead, we have to find the
                        // biggest multiple only, which is defined by
                        // epsilon_prime_squared * P_pre *
                        // Math.floor(e(C^{i,k})/epsilon_prime_squared/P_pre) as
                        // can be seen centrally in the left column on page
                        // 1534. This lets us get rid of the while lopp
                        // entirely.
                        while end < cut {
                            let p_cut = last_cut - cut;

                            stack.push(GeneralizedConfiguration {
                                configuration: Configuration {
                                    processing_time: p_cut,
                                    ..c.0.configuration.clone()
                                },
                                window: Rc::from(Window {
                                    resource_amount: r,
                                    machine_count: c.0.window.machine_count,
                                }),
                            });

                            cuts_done += 1.0;
                            last_cut = cut;
                            cut -= step_width;
                        }

                        stack.push(GeneralizedConfiguration {
                            configuration: Configuration {
                                processing_time: last_cut - end,
                                ..c.0.configuration.clone()
                            },
                            window: Rc::clone(&c.0.window),
                        });

                        (
                            // for subsequent configs, we use the current resource amount
                            c.0.window.resource_amount,
                            // we usually decrement k, but for large configs we must make several steps at once
                            cuts_done,
                        )
                    } else {
                        // we are not cutting the configuration, so we just use the resource amount from the last cut
                        stack.push(GeneralizedConfiguration {
                            configuration: c.0.configuration.clone(),
                            window: Rc::from(Window {
                                resource_amount: r,
                                machine_count: c.0.window.machine_count,
                            }),
                        });
                        (r, 0.0)
                    };

                    (stack, p - processing_time, r, k - cuts_done)
                },
            );

            // TODO: investigate how the length can be eps'*P_pre
            stack.push(GeneralizedConfiguration {
                configuration: Configuration::empty(),
                window: Rc::from(Window {
                    resource_amount: problem.resource_limit,
                    machine_count: problem.machine_count,
                }),
            });
            stacks.push(stack);
            stacks
        });

    for (i, stack) in stacks.iter().enumerate() {
        println!("  --- K_{} ---  ", i + 1);
        for config in stack {
            println!("{:?}", config);
        }
    }

    stacks
}

fn assign_narrow_jobs(_x_tilde: &GeneralizedSelection, _y_tilde: &NarrowJobSelection) {
    // TODO: fix the above FIXME and base the following impl on the above one
    // for x_tilde

    // See page 1534 left bottom
    // for job j in narrow_jobs
    //   for config c in K_(i,k+1) # configurations between C(i,k) and C(i,k+1)
    //     y_bar_(j,w_(i,k)) += P_pre(C)/P_pre(w(C)) * y_tilde_(j,w(C))

    // do the things on page 1534 right top

    // return y_bar
}
