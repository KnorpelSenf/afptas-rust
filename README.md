# afptas-rust

## Introduction

This is the implementation of the first algortihm of the paper **Approximation schemes for machine scheduling with resource (in-)dependent processing times** by Jansen, Maack, and Rau.
It can solve scheduling problems with resource constraint in polynomial time (less than cubic) in the number of jobs.
The prior statement simplifies things to the point of obnoxiousness, please read the paper.

DOI: <https://doi.org/10.1145/3302250>

This repository provides a CLI program that can read in problem instances from CSV files, and compute a valid schedule with resource constraints.
The CSV file is expected to have the following structure.

```csv
processing_time,resouce_amount
1.3,4.2
1.0,5
7,7
```

The processing time of each job has to be the column on the left.
The resouces amount of each job has to be the column on the right.
The column headers are ignored.

By default, the computed schedule is printed to stdout.
An optional SVG rendering step is included.

## Installation

This CLI is written in Rust.
We do not provide binaries, so you need to compile the program yourself.

You need to [install Rust](https://www.rust-lang.org/tools/install).
Run `cargo --version` to confirm that cargo was installed successfully.
We used version `1.80.1`.

This program relies on [CBC](https://github.com/rust-or/good_lp#cbc) to solve ILPs.
It therefore requires you to have it installed on your system.

On linux, run

```sh
sudo apt update
sudo apt install coinor-cbc coinor-libcbc-dev
```

to install it.

On MacOS, run

```sh
brew install cbc
```

instead.

On Windows, install WSL2 and goto Linux.

## Usage

You can run the program with `cargo run`.
Addition arguments need to be separated via `--`.

For instance, you can inspect all program arguments of this CLI by running `cargo run -- --help`.

```sh
$ cargo run -- --help
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.01s
     Running `target/debug/afptas-rust --help`
Usage: afptas-rust [OPTIONS] --epsilon <EPSILON> --machines <MACHINES> --resource-limit <RESOURCE_LIMIT> --job-file <JOB_FILE>

Options:
  -e, --epsilon <EPSILON>
          Solution accuracy
  -m, --machines <MACHINES>
          Number of machines
  -r, --resource-limit <RESOURCE_LIMIT>
          Resource limit
  -j, --job-file <JOB_FILE>
          Input CSV file containing jobs in the format "processing_time,resource_amount"
      --svg
          Render the schedule to an SVG file in the directory "schedules"
      --open
          Open the rendered SVG if created
  -h, --help
          Print help
  -V, --version
          Print version
```

For example, here is how you can compute a schedule for the provided problem instance with 1000 jobs, render the solution to an SVG, and open it in the default system PDF viewer.

```sh
cargo run -- -e 0.5 -m 10 -r 2 -j jobs_1_000.csv --svg --open
```

It sets the following values:

- epsilon: 0.5
- number of machines: 10
- resource constraint: 2

## Known Example Instances

We provide 9 pre-computed example sets of jobs.

A few interesting combinations of input parameters are listed in the below table.

| Job File         | Epsilon | Machines | Resources |
| ---------------- | ------- | -------- | --------- |
| `jobs_1_000.csv` | `0.5`   | `5`      | `3`       |
| `jobs_1_000.csv` | `0.5`   | `10`     | `2`       |
| `jobs_3_000.csv` | `0.5`   | `10`     | `2`       |

## Using Release Builds

By default, the CLI is run in debug mode.
It is very slow.

However, given that we delegate all meaningful processing to the ILP solver, this does not really matter much.
The only affected parts are the generalization step, and the last step to reduce resouce amounts.
They typically complete in a few seconds.

If you still want to make this run faster, pass `--release`.

```sh
cargo run --releasea -- # ...
```

## Generating Problem Instances

We furthermore provide a script to generate large amounts of jobs.
It is limited to generating pairs of processing time and resource amount.
Both values are picked using a linear random distribution in two indepently configurable intervals.

```sh
# Setup (once)
python -m venv venv
source venv/bin/activate

# Generating jobs files
python gen.py --help
```

The last command will output the following usage instructions.
The should be self-explanatory.

```sh
usage: job generation util [-h] [-o FILENAME] [-n COUNT] [--min-processing-time MIN_PROCESSING_TIME] [--max-processing-time MAX_PROCESSING_TIME] [--min-resource-amount MIN_RESOURCE_AMOUNT] [--max-resource-amount MAX_RESOURCE_AMOUNT]

generate very large random scheduling instances

options:
  -h, --help            show this help message and exit
  -o FILENAME, --filename FILENAME
  -n COUNT, --count COUNT
  --min-processing-time MIN_PROCESSING_TIME
  --max-processing-time MAX_PROCESSING_TIME
  --min-resource-amount MIN_RESOURCE_AMOUNT
  --max-resource-amount MAX_RESOURCE_AMOUNT
```

## Setting Log Levels

By default, the CLI only outputs a few progress bars.

If you want to debug the program, you can tell the CLI to output more detailed logs via the `RUST_LOG` environment variable.
The possible values are listed below.

```sh
export RUST_LOG=error # (currently unused)
export RUST_LOG=warn # (currently unused)
export RUST_LOG=info # prints basic info about the problem instance and measures computation time
export RUST_LOG=debug # provides detailed logs about individual processing steps, disables progress bar
export RUST_LOG=trace # cranks out ~1G of logging data when solving instances with 5000+ jobs
```

## Further Information

The main entrypoint of the CLI is in `src/main.rs`.

The implementation of the paper happens entirely in `src/algo.rs`.
Check it out.
