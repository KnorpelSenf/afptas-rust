use std::cmp::Ordering;

#[derive(Debug)]
pub struct InputData {
    pub epsilon: f64,
    pub instance: Instance,
}

#[derive(Debug)]
pub struct Instance {
    pub machine_count: u32,
    pub resource_limit: f64,
    pub jobs: Box<Vec<Job>>,
}

#[derive(Debug)]
pub struct Job {
    pub id: u32,
    pub processing_time: f64,
    pub resource_amount: f64,
}
#[derive(Debug)]
pub struct Schedule {
    pub mapping: Box<Vec<JobPosition>>,
}

#[derive(Debug)]
pub struct JobPosition {
    pub machine: u32,
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
    if 1.0 / epsilon >= machine_count.into() {
        unimplemented!("second case");
    }
    let _n = jobs.len();
    let epsilon_prime = epsilon / 5.0;
    let _p_max = jobs
        .iter()
        .max_by(compare_resource_amount)
        .expect("no jobs found");
    let threshold = epsilon_prime * resource_limit;
    let (narrow_jobs, mut wide_jobs) = jobs
        .iter()
        .partition::<Vec<_>, _>(|job| job.resource_amount < threshold);
    wide_jobs.sort_by(compare_processing_time);
    println!("Wide {:?}", wide_jobs);
    println!("Narrow {:?}", narrow_jobs);
    Schedule {
        mapping: Box::from(vec![]),
    }
}

fn compare_processing_time(job0: &&Job, job1: &&Job) -> Ordering {
    job0.processing_time
        .partial_cmp(&job1.processing_time)
        .expect("invalid processing time")
}

fn compare_resource_amount(job0: &&Job, job1: &&Job) -> Ordering {
    job0.resource_amount
        .partial_cmp(&job1.resource_amount)
        .expect("invalid resource amount")
}
