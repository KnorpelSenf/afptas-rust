use svg::node::element::{Group, LinearGradient, Rectangle, Stop, Style, Text};
use svg::node::Text as SvgText;
use svg::Document;

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

pub fn svg(schedule: Schedule) -> String {
    // Create the linear gradient for the background
    let gradient = LinearGradient::new()
        .set("id", "background")
        .set("y1", "0")
        .set("y2", "1")
        .set("x1", "0")
        .set("x2", "0")
        .add(Stop::new().set("stop-color", "#eeeeee").set("offset", "5%"))
        .add(
            Stop::new()
                .set("stop-color", "#b0b0ee")
                .set("offset", "95%"),
        );

    // Create the SVG document
    let document = Document::new()
        .set("width", "100%")
        .set("height", "100%")
        .set("version", "1.1")
        .set("xmlns", "http://www.w3.org/2000/svg")
        .set("xmlns:svg", "http://www.w3.org/2000/svg")
        .add(gradient)
        .add(Style::new(
            r#"
            text { font-family:monospace; font-size:12px; fill:black; }
            #title { text-anchor:middle; font-size:20px; }
            .machine-box { fill:blue; stroke-width:1; stroke:black; }
            .machine-label { text-anchor:middle; dominant-baseline:middle; font-size:15px; }
        "#,
        ))
        .add(
            Rectangle::new()
                .set("x", 0)
                .set("y", 0)
                .set("width", "100%")
                .set("height", "100%")
                .set("fill", "url(#background)"),
        )
        .add(create_title())
        .add(create_machine("Machine A", 10, 10))
        .add(create_machine("Machine B", 10, 21))
        .add(create_machine("Machine C", 10, 32));

    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="no"?>
{}"#,
        document.to_string()
    )
}

fn create_title() -> Text {
    Text::new("Schedule")
        .set("id", "title")
        .set("x", "50%")
        .set("y", 24)
        .set("fill", "rgb(0,0,0)")
}

fn create_machine(name: &str, x: usize, y: usize) -> Group {
    let machine_box = Rectangle::new()
        .set("x", format!("{}%", x))
        .set("y", format!("{}%", y))
        .set("width", "10%")
        .set("height", "10%")
        .set("class", "machine-box");

    let machine_label = Text::new(name)
        .set("x", format!("{}%", x + 5)) // Centered on the rectangle
        .set("y", format!("{}%", y + 5)) // Centered on the rectangle
        .set("class", "machine-label");

    Group::new().add(machine_box).add(machine_label)
}
