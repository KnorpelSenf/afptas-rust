use std::{cmp::Ordering, collections::HashMap};

use crate::max_min::max_min;

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
        self.resource_amount < threshold
    }
}
impl PartialEq for Job {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Job {}

#[derive(Debug)]
pub struct Configuration {
    index: HashMap<i32, usize>, // job.id -> index in vec
    pub jobs: Box<Vec<(Job, i32)>>,
}
impl Configuration {
    pub fn get(&self, job: Job) -> Option<i32> {
        Some(self.jobs[*self.index.get(&job.id)?].1)
    }
    pub fn set(&mut self, job: Job, count: i32) {
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
    pub fn machines(&self) -> i32 {
        self.jobs.iter().map(|pair| pair.1).sum()
    }
    pub fn processing_time(&self) -> f64 {
        self.jobs
            .iter()
            .map(|pair| pair.1 as f64 * pair.0.processing_time)
            .sum()
    }
    pub fn resource_amount(&self) -> f64 {
        self.jobs
            .iter()
            .map(|pair| pair.1 as f64 * pair.0.resource_amount)
            .sum()
    }
    pub fn is_valid(&self, instance: Instance) -> bool {
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
        unimplemented!("second case");
    }
    let n = jobs.len();
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

    println!("Wide {:?}", wide_jobs);
    println!("Narrow {:?}", narrow_jobs);
    let p_w: f64 = wide_jobs.iter().map(|job| job.processing_time).sum();
    let i_sup = create_i_sup(epsilon_prime_squared, p_w, wide_jobs);

    let _ = max_min(
        n,
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
    println!("Adding {} jobs to generate I_sup", additional_jobs.len());
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
