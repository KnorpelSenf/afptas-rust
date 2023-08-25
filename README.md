# afptas-rust

This is a pseudo version of the algorithm that reflects our current understanding of the problem.

```pseudo
fn afptas(I, epsilon):
  let (J,m,R) = I
  let n = length(J)
  let (j0,p0,r0), ..., (jn,pn,rn) = J
  
# (i)
  let epsilon' = epsilon / 5
  let p_max = max(p0, ..., pn)
  
  let I_groups = linear_grouping(I)
  let (widest, ...I_sup) = I_groups
  assert length_wide_jobs(I_sup) <= 1 / (epsilon' ** 2)

# (ii)
  let x_pre = preemptive_schedule(I_sup)
  assert makespan(I_sup) <= (1 + epsilon') * OPT_pre

# (iii)
  let x_gen = generalize(x_pre)
  let (config, window) = x_gen
  let (wr, wm) = window

# (iv)
  group_somehow()

# (v)
  let ((x', y'), LPw) = solve(I_sup)
  assert zeros(LPw) <= 4 + 3 / (epsilon' ** 2) + length(Jn)
  assert lengthened_factor(LPw) <= 1 + epsilon'

# (vi)
  let I_int = integral_schedule(x', y', I_sup)
  assert schedule is not lengthened by more than something

# (vii)
  let result = append_greedily(I_int, widest)
  return result

fn preemptive_schedule(I):
  # todo (involves LP)

fn generalize(S):
  # todo (involves ILP)
```
