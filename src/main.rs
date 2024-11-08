mod algo;
mod in_data;
mod pretty;

use std::fs::{create_dir_all, File};
use std::io::Write;
use std::time::Instant;

use algo::{InstanceJob, Job, Schedule};

use crate::algo::{compute_schedule, Instance};
use crate::in_data::parse;
use crate::pretty::{display, pretty, svg};

fn main() {
    let (to_svg, instance) = parse();

    let Instance {
        epsilon,
        machine_count,
        resource_limit,
        ref jobs,
    } = instance;
    let job_count = jobs.len();
    println!("Scheduling {job_count} jobs on {machine_count} machines with a resource limit of {resource_limit} with epsilon={epsilon} close to OPT");

    let job_backup = jobs.clone();
    let start = Instant::now();
    let schedule = compute_schedule(instance);
    let duration = start.elapsed();

    println!("Asserting that all jobs are scheduled");
    compare_jobs_before_and_after_schedule(job_backup, schedule.clone());

    println!("Done in {:?}.", duration);
    if to_svg {
        let file_data = svg(resource_limit, schedule);
        create_dir_all("./schedules/").expect("cannot create directory ./schedules");
        let path = format!(
            "schedules/schedule_m-{machine_count}_eps-{epsilon}_res-{resource_limit}_jobs-{job_count}.svg"
        );
        let mut file = File::create(path.clone()).expect("cannot create file schedule.svg");
        file.write_all(file_data.as_bytes())
            .expect(&format!("cannot write to file {path}"));
        println!("Result is written to {path}");
    } else if job_count <= 1000 {
        println!("Result is (prettified):");
        println!("{}", pretty(schedule));
    } else {
        println!("Result is:");
        println!("{}", display(schedule));
        println!("{machine_count} machines.");
    }
}

fn compare_jobs_before_and_after_schedule(mut jobs: Vec<InstanceJob>, schedule: Schedule) {
    let mut scheduled_jobs: Vec<Job> = schedule
        .chunks
        .into_iter()
        .flat_map(|chunk| {
            chunk
                .machines
                .into_iter()
                .flat_map(|machine| machine.jobs.into_iter())
        })
        .collect();

    assert!(
        jobs.len() == scheduled_jobs.len(),
        "number of jobs was {} but the schedule contained {} jobs",
        jobs.len(),
        scheduled_jobs.len(),
    );

    jobs.sort_by(|job0, job1| {
        job0.resource_amount
            .partial_cmp(&job1.resource_amount)
            .expect("cannot compare bad processing time")
    });
    jobs.sort_by(|job0, job1| {
        job0.processing_time
            .partial_cmp(&job1.processing_time)
            .expect("cannot compare bad processing time")
    });

    scheduled_jobs.sort_by(|job0, job1| {
        job0.resource_amount
            .partial_cmp(&job1.resource_amount)
            .expect("cannot compare bad processing time")
    });
    scheduled_jobs.sort_by(|job0, job1| {
        job0.processing_time
            .partial_cmp(&job1.processing_time)
            .expect("cannot compare bad processing time")
    });

    assert!(
        jobs.iter()
            .zip(scheduled_jobs)
            .all(
                |(job, scheduled_job)| job.resource_amount == scheduled_job.resource_amount
                    && job.processing_time == scheduled_job.processing_time
            ),
        "scheduled jobs differ in resource amount or processing time!"
    );
}
