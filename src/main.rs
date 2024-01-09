use std::io::BufRead;
use std::io::stdin;
use std::env::args;

fn main() {
    // get the first 3 command line arguments, which are numbers and than read all input lines with read_line inside a for loop and save the result in an array
    let (m, R, e) = (args().nth(1).unwrap().parse::<i32>().unwrap(), args().nth(2).unwrap().parse::<i32>().unwrap(), args().nth(3).unwrap().parse::<i32>().unwrap());
    let mut jobs = Vec::new();
    for line in stdin().lock().lines() {
        let line = line.unwrap();
        let mut line = line.split(",");
        let (x, y) = (line.next().unwrap().parse::<i32>().unwrap(), line.next().unwrap().parse::<i32>().unwrap());
        jobs.push((x, y));
    }

    println!("{} {} {}", a, b, c);
    for (x, y) in jobs {
        println!("{} {}", x, y);
    }
}
