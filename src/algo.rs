#[derive(Debug)]
pub struct InputData {
    pub epsilon: f64,
    pub instance: Instance,
}

#[derive(Debug)]
pub struct Instance {
    pub machine_count: u32,
    pub resource_limit: f64,
    pub jobs: Box<[Job]>,
}

#[derive(Debug)]
pub struct Job {
    pub id: u32,
    pub processing_time: f64,
    pub resource_amount: f64,
}
#[derive(Debug)]
pub struct Schedule {
    pub mapping: Box<[JobPosition]>,
}

#[derive(Debug)]
pub struct JobPosition {
    pub machine: u32,
    pub starting_time: f64,
}

pub fn compute_schedule(_in_data: InputData) -> Schedule {
    Schedule {
        mapping: Box::from([]),
    }
}
