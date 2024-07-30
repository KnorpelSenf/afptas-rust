use crate::algo::Schedule;

use std::{cmp::max, iter::repeat};

const TICK: f64 = 0.5;

pub fn pretty(schedule: Schedule) -> String {
    let machine_count = schedule.mapping.len();
    if machine_count == 0 {
        return String::from("<empty schedule>");
    }
    let job_count: usize = schedule.mapping.iter().map(|s| s.jobs.len()).sum();
    let label_width = (max(job_count, machine_count) - 1).to_string().len();
    let column_width = 2 + label_width;
    let mut columns: Vec<_> = schedule
        .mapping
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let m = i + 1;
            s.jobs.iter().fold(
                vec![format!("-{:->label_width$}-", m), " ".repeat(column_width)],
                |mut agg, job| {
                    let id = job.id;
                    let mut processing_time = job.processing_time;
                    // FIXME: assert!(processing_time >= TICK, "job {id} too small to be printed!");
                    println!("job {id} too small to be printed!");
                    agg.push(format!("-{:->label_width$}-", id));
                    processing_time -= TICK;
                    while processing_time >= TICK {
                        agg.push(format!(" {:>label_width$} ", id));
                        processing_time -= TICK;
                    }
                    agg
                },
            )
        })
        .collect();
    let row_count = columns.iter().map(|c| c.len()).max().unwrap();
    for col in columns.iter_mut() {
        col.extend(repeat(" ".repeat(column_width)).take(row_count - col.len()));
    }
    let footer = format!("Scheduled {job_count} jobs on {machine_count} machines");
    let mut result =
        String::with_capacity(row_count * (2 + machine_count * (1 + column_width)) + footer.len());
    for i in 0..row_count {
        result.push_str("|");
        for j in 0..machine_count {
            result.push_str(&columns[j][i]);
            result.push_str("|");
        }
        result.push_str("\n");
    }
    result.push_str(&footer);
    result
}
