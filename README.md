# afptas-rust

## Setup

1. You need to install Rust | <https://www.rust-lang.org/tools/install>
2. You need to install an LP solver by running `sudo apt install coinor-cbc coinor-libcbc-dev` (Linux) or `brew install cbc` (Mac) | <https://github.com/rust-or/good_lp#cbc>

## Iteration I

This is a pseudo version of the algorithm that reflects our current
understanding of the problem.

- [Algorithm in Pseudo](./pseudo.md)

Progress:

- [x] (i)
- [x] (ii)
- [x] (iii)
- [x] (iv)
- [x] (v)
- [x] (vi)
- [x] (vii)

## Iteration II

- [List of symbols](./symbols.md)
- [List of graphical illustrations](./illustrations.md)

Progress:

- [x] (i)
- [x] (ii)
- [x] (iii)
- [x] (iv)
- [x] (v)
- [x] (vi)
- [x] (vii)

## Iteration III

Run it via

```sh
cargo run -- -e 0.5 -m 3 -r 11 -j jobs.csv
```
