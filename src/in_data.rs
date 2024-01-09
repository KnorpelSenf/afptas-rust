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
