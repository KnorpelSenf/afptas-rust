use good_lp::{
    constraint, default_solver, variable, variables, Expression, ProblemVariables, Solution,
    SolverModel, Variable,
};

use crate::algo::Job;

// pub fn example() -> Result<(), Box<dyn Error>> {
//     variables! {
//         vars:
//           a <= 4;
//           0 <= b (integer) <= 1;
//     } // variables can also be added dynamically
//     let solution = vars
//         .maximise(a + b)
//         .using(default_solver) // multiple solvers available
//         .solve()?;
//     println!("a={}   b={}", solution.value(a), solution.value(b));
//     println!("a + b = {}", solution.eval(a + b));
//     Ok(())
// }

// pub fn other_example() -> Result<(), Box<dyn Error>> {
//     // Create variables in a readable format with a macro...
//     variables! {
//        vars:
//            a <= 1;
//            2 <= b <= 4;
//     }
//     // ... or add variables programmatically
//     vars.add(variable().min(2).max(9));

//     let solution = vars
//         .maximise(10 * (a - b / 5) - b)
//         .using(default_solver)
//         .with(a + 2. << b) // or (a + 2).leq(b)
//         .with(1 + a >> 4. - b)
//         // .with(constraint!(1 + a >= 4. - b))
//         .solve()?;

//     assert_eq!(solution.value(a), 1.);
//     assert_eq!(solution.value(b), 3.);
//     Ok(())
// }

pub fn solve_ilp(
    q: Vec<f64>,
    jobs: &Vec<Job>,
    narrow_threshold: f64,
    wide_job_max_count: i32, // should be 1 / epsilon_prime
    machine_count: i32,
    resource_limit: f64,
) {
    let max = wide_job_max_count as f64;
    let mut prob = Ilp::new(machine_count, resource_limit);

    let variables: Vec<_> = q
        .iter()
        .zip(jobs)
        .map(|q_job| {
            prob.add(
                ConfigurationCandidate {
                    q: *q_job.0,
                    p: q_job.1.processing_time,
                    r: q_job.1.resource_amount,
                    a: 0,
                },
                if q_job.1.is_wide(narrow_threshold) {
                    max
                } else {
                    1.
                },
            )
        })
        .collect();

    let solution = prob.find_configuration();

    let a_star: Vec<_> = variables.iter().map(|&v| solution.value(v)).collect();

    for i in a_star {
        println!("{}", i);
    }
}

struct ConfigurationCandidate {
    q: f64,
    p: f64,
    r: f64,
    a: i32,
}

struct Ilp {
    vars: ProblemVariables,
    machine_count: Expression,
    resource_amount: Expression,
    machine_limit: i32,
    resource_limit: f64,
    objective: Expression,
}

impl Ilp {
    fn new(machine_limit: i32, resource_limit: f64) -> Ilp {
        Ilp {
            vars: variables!(),
            machine_limit,
            resource_limit,
            machine_count: 0.into(),
            resource_amount: 0.into(),
            objective: 0.into(),
        }
    }

    fn add(&mut self, job: ConfigurationCandidate, max: f64) -> Variable {
        let v = self.vars.add(variable().integer().min(0).max(max));
        self.objective += job.q * (job.a as f64) / job.p;
        self.machine_count += job.a;
        self.resource_limit += job.a as f64 * job.r;
        v
    }

    fn find_configuration(self) -> impl Solution {
        self.vars
            .maximise(self.objective)
            .using(default_solver)
            .with(constraint!(self.machine_count <= self.machine_limit))
            .with(constraint!(self.resource_amount <= self.resource_limit))
            .solve()
            .expect("no solution")
    }
}
