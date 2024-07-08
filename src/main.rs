mod algo;
mod in_data;
mod pretty;

use algo::{Job, MachineSchedule, Schedule};

use crate::algo::{compute_schedule, Instance};
use crate::in_data::parse;
use crate::pretty::pretty;

fn main() {
    let test_schedules = vec![
        Schedule { mapping: vec![] },
        Schedule {
            mapping: vec![MachineSchedule { jobs: vec![] }],
        },
        Schedule {
            mapping: vec![
                MachineSchedule { jobs: vec![] },
                MachineSchedule { jobs: vec![] },
                MachineSchedule { jobs: vec![] },
            ],
        },
        Schedule {
            mapping: vec![
                MachineSchedule { jobs: vec![] },
                MachineSchedule {
                    jobs: vec![
                        Job {
                            id: 0,
                            processing_time: 1.5,
                            resource_amount: 0.5,
                        },
                        Job {
                            id: 1,
                            processing_time: 0.5,
                            resource_amount: 0.5,
                        },
                        Job {
                            id: 2,
                            processing_time: 2.5,
                            resource_amount: 0.5,
                        },
                    ],
                },
                MachineSchedule { jobs: vec![] },
            ],
        },
        Schedule {
            mapping: vec![
                MachineSchedule {
                    jobs: vec![
                        Job {
                            id: 0,
                            processing_time: 1.5,
                            resource_amount: 0.5,
                        },
                        Job {
                            id: 1,
                            processing_time: 0.5,
                            resource_amount: 0.5,
                        },
                        Job {
                            id: 2,
                            processing_time: 2.5,
                            resource_amount: 0.5,
                        },
                    ],
                },
                MachineSchedule {
                    jobs: vec![
                        Job {
                            id: 3,
                            processing_time: 0.5,
                            resource_amount: 0.5,
                        },
                        Job {
                            id: 4,
                            processing_time: 1.5,
                            resource_amount: 0.5,
                        },
                        Job {
                            id: 5,
                            processing_time: 1.5,
                            resource_amount: 0.5,
                        },
                    ],
                },
                MachineSchedule {
                    jobs: vec![
                        Job {
                            id: 6,
                            processing_time: 0.5,
                            resource_amount: 0.5,
                        },
                        Job {
                            id: 7,
                            processing_time: 0.5,
                            resource_amount: 0.5,
                        },
                        Job {
                            id: 8,
                            processing_time: 1.0,
                            resource_amount: 0.5,
                        },
                    ],
                },
                MachineSchedule {
                    jobs: vec![
                        Job {
                            id: 9,
                            processing_time: 1.0,
                            resource_amount: 0.5,
                        },
                        Job {
                            id: 10,
                            processing_time: 1.0,
                            resource_amount: 0.5,
                        },
                        Job {
                            id: 11,
                            processing_time: 1.5,
                            resource_amount: 0.5,
                        },
                    ],
                },
            ],
        },
    ];
    for s in test_schedules {
        let prettified = pretty(s);
        println!("{prettified}",);
    }
    return;

    let instance = parse();

    let Instance {
        epsilon,
        machine_count,
        resource_limit,
        ref jobs,
    } = instance;
    println!("Scheduling {} jobs on {} machines with a resource limit of {} with epsilon={} close to OPT", jobs.len(), machine_count, resource_limit, epsilon);

    let schedule = compute_schedule(instance);

    println!("Done, result is:\n{}", pretty(schedule));
}
