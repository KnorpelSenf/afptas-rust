use crate::algo::{Schedule, ScheduleChunk};

use std::{cmp::max, iter::repeat};

const TICK: f64 = 0.5;

pub fn pretty(schedule: Schedule) -> String {
    schedule
        .chunks
        .into_iter()
        .map(|chunk| pretty_chunk(chunk))
        .collect()
}

fn pretty_chunk(chunk: ScheduleChunk) -> String {
    let machine_count = chunk.machines.len();
    if machine_count == 0 {
        return String::from("<empty schedule>");
    }
    let job_count: usize = chunk.machines.iter().map(|s| s.jobs.len()).sum();
    let label_width = (max(job_count, machine_count) - 1).to_string().len();
    let column_width = 2 + label_width;
    let mut columns: Vec<_> = chunk
        .machines
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
    let mut result = String::with_capacity(row_count * (2 + machine_count * (1 + column_width)));
    for i in 0..row_count {
        result.push_str("|");
        for j in 0..machine_count {
            result.push_str(&columns[j][i]);
            result.push_str("|");
        }
        result.push_str("\n");
    }
    result
}

pub fn display(schedule: Schedule) -> String {
    schedule
        .chunks
        .into_iter()
        .map(|chunk| display_chunk(chunk))
        .collect()
}
fn display_chunk(chunks: ScheduleChunk) -> String {
    let machine_count = chunks.machines.len();
    let job_count: usize = chunks.machines.iter().map(|m| m.jobs.len()).sum();
    let mut str = String::with_capacity((job_count as f64).log10() as usize * job_count);

    for machine in chunks.machines {
        str.push_str(&format!("M{}: ", machine_count));
        for job in machine.jobs {
            str.push_str(&format!("{} ", job.id));
        }
        str.push_str("\n");
    }

    str
}
