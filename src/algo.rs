use crate::in_data::InputData;

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
