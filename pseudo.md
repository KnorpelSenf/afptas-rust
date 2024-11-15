# Pseudo Implementation

This is a pseudo version of the algorithm that reflects our current understanding of the problem.

```pseudo
fn afptas(I, epsilon):
  let (J,m,R) = I
  let n = length(J)
  let (j0,p0,r0), ..., (jn,pn,rn) = J

# (i)
  let epsilon' = epsilon / 5
  let p_max = max(p0, ..., pn)

  let I_groups = linear_grouping(J, epsilon, R)
  let (widest, ...I_sup) = I_groups
  assert length_wide_jobs(I_sup) <= 1 / (epsilon' ** 2)

# (ii)
  # obtain a sparse bitvector (?) defining a configuration
  let x_pre = preemptive_schedule(I_sup)
  assert makespan(I_sup) <= (1 + epsilon') * OPT_pre

# (iii)
  let (Wpre, x~, y~) = generalize(C_I', C_pre, C_W, x_pre)

# (iv)
  W' = group_by_resources(Wpre)

# (v)
  let (x', y') = match_with_windows(W', x~, y~)

# (vi)
  let I_int = integral_schedule(x', y')
  assert schedule is not lengthened by more than something

# (vii)
  let result = append_greedily(I_int, widest)
  return result

fn linear_grouping(J, epsilon', R):
  let threshold = epsilon' * R
  let Jn = { j for (j, p, r) in J if r >= threshold }
  let Jw = { j for (j, p, r) in J if r <  threshold }

  # intepret resource consumption as width and
  # processing time as height
  return strip_packing(Jw)

fn strip_packing((w0,h0), ..., (wn,hn)):
  # sorting by w
  # let stack []
  # loop over all j
    # place on stack
  # Pw = length(stack)
  # perform linear grouping as in paper on page 1525, paragraph 1, left side


fn preemptive_schedule(I):
  # find a minimal subset of C_I that satisfies 2.3, 2.4, 2.5
  # and return this subset
  max_min_resource_sharing(I)

fn max_min_resource_sharing(I):
  let (J,_,_) = I
  let C_I = all_valid_possible_configurations(I)
  let M = length(J)

  let B = {x for x in R^length(C_I) if norm_1(x) == 1}

  # B is a [0,1]^n block of values that weight the different configurations.
  # We now want to find x in B which adheres to the constraints and is optimal.
  # We do this using Grigoriadis et al. and
  # need to solve the block-problem along the way and
  # need to solve an ILP along that way.
  # Idea: take xv, compute cost q, solve ILP to get x^, adjust xv, repeat

fn frac_job_of_schedule(j, x, C_I):
  let (_, p, _) = j
  let sum = 0
  for (let i = 0; i < length(C_I); i++):
    sum += num_contained_Cj(C_I[i]) * x[i] / p
  return sum

fn num_contained_Cj(C):
  # return how many times j is in C

# C_I'  -- all configurations
# Cpre  -- configurations from solution, corresponds with x_pre
# C_W   -- all configurations of wide jobs, i.e. C|Jw
# x_pre -- solution to previous LP
fn generalize(C_I', Cpre, C_W, x_pre):
  # lemma 2.2
  let C_preW = { C|Jw for C in C_I' if processing_time(C, x_pre) > 0 }
  Wpre = { w for (_, w) in C_preW }

  # x~ is a vector that maps a
  # generalized configuration -> sum of processing times of configurations where the wide jobs look similar
  x~ = []
  for C in C_W:
    sum = 0
    for C' in Cpre:
      if C'|Jw == C
        # combine C with window from Wpre
        main_window = max(Wpre)
        sum += processing_time(C)
    x~[C] = sum

  # y~ is a matrix that maps
  # (a narrow job, a main window) -> how much of the job can be processed in the window
  y' = [][]
  for j in J_N:
    for w in W:
      sum = 0
      for C in Cpre:
        _, wc = C|J_W
        if w == wc:
          sum += C(j) * processing_time(C)
      y'[j][w] = sum
   y~ = y'.flat()

  return Wpre, x~, y~

fn group_by_resources(Wpre):
  # group stuff as described around Figure 2

fn match_with_windows(W', x~, y~):
  # compute x', y' as described at end of proof for lemma 2.3
  return x', y'

fn integral_schedule(x', y'):
  # perform the operations as described in lemma 2.4

fn append_greedily(I_int, widest):
  I_int.push(...widest)
  return I_int
```
