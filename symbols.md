# List of Symbols

| Markdown | Symbol                                                                                                                                                                       | Location                                | Description                                                                                                                                                                                 |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| a*       | $a^*$                                                                                                                                                                        | Lemma 2.1                               | Solution to $\text{ILP}_{kKP}$                                                                                                                                                              |
| B        | $B = \{x\in\mathbb{R}^{\mid C_I\mid}_{\geq 0}\mid \sum\limits_{C\in C_I}x_C=1\}$                                                                                             | Lemma 2.1                               | A nonempty convex compact set, the set of selections of valid configurations with overall makespan $1=OPT_\text{pre}(I)$                                                                    |
| C        | $C\in\mathbb{N}^\mathcal{J}$                                                                                                                                                 | Preemptive Schedule                     | Configuration, a multiset of jobs                                                                                                                                                           |
| CI       | $C_I$                                                                                                                                                                        | Preemptive Schedule                     | Set of all valid configurations of $I$                                                                                                                                                      |
| CJW      | $C\|_{\mathcal{J}_W}$                                                                                                                                                        | Handling the narrow jobs                | Configuration consisting of all wide jobs in $C$                                                                                                                                            |
| Cpre     | $\mathcal{C}_\text{pre}=\{C\in\mathcal{C}_{I^\prime}\mid(x_\text{pre})_C>0\}$                                                                                                | Remark 2.1                              | Set of configurations used in $x_\text{pre}$                                                                                                                                                |
| CpreW    | $\mathcal{C}_{\text{pre},W}=(\mathcal{C}_\text{pre})_W$                                                                                                                      | n/a                                     | Set of valid configurations used in $x_\text{pre}$ reduced to the wide jobs                                                                                                                 |
| CW       | $C_W=\{C\|{\mathcal{J}_W}\mid C\in\mathcal{C}\}$ for $\mathcal{C}\subseteq C_I$                                                                                              | Handling the narrow jobs                | Subset of valid configurations reduced to the wide jobs                                                                                                                                     |
| C*       | $C^*\in\mathcal{C}$                                                                                                                                                          | Lemma 2.1                               | Comprised of many $C^\ast(j)$, will be added to the solution of the min-max resource sharing problem                                                                                        |
| Cij      | $C_{i,j}\in K_i\subset\mathcal{C}_W$                                                                                                                                         | Lemma 2.3                               | Configuration $j$ in $K_i$ with $R(C_{i,j-1})\leq R(C_{i,j})\leq R(C_{i,j+1})$                                                                                                              |
| C(j)     | $C(j)$                                                                                                                                                                       | Preemptive Schedule                     | Number of times the job $j$ is contained in the configuration $C$ $$\\$$ Note: $C(j)\in\{0,1\}$ if $j\in\mathcal{J}_N$ and $C(j)\in\{0,\dots,1/\varepsilon^\prime\}$ if $j\in\mathcal{J}_W$ |
| C^ik     | $C^{i,k}\in K_i$ for $k\in\{1,\dots,k_i-1\}$                                                                                                                                 | Lemma 2.3                               | Configuration in $K_i$ intersecting with $k\varepsilon^{\prime 2}P_\text{pre}$, meaning $s(C)<k\varepsilon^{\prime 2}P_\text{pre}\leq e(C)$                                                 |
| C*(j)    | $C^\ast(j) =a^\ast_j$ for $j\in\mathcal{J}$                                                                                                                                  | Lemma 2.1                               | Number of times the job $j$ is contained in the configuration $C^\ast$                                                                                                                      |
| (C, w)   | $(C, w)$                                                                                                                                                                     | Handling the narrow jobs                | Generalized configuration                                                                                                                                                                   |
| delta    | $\delta\in\mathcal{O}(\rho)$                                                                                                                                                 | Lemma 2.1                               | Accuracy of solution to $\text{ILP}_{kKP}$                                                                                                                                                  |
| e        | $e\in\{1\}^M\subset\mathbb{R}^M$                                                                                                                                             | Lemma 2.1                               | Vector with $M$ $1$'s                                                                                                                                                                       |
| e(Cij)   | $e(C_{i,j})=s(C_{i,j+1})$                                                                                                                                                    | Lemma 2.3                               | End position of configuration $j$ in $K_i$, same as next start position                                                                                                                     |
| ε        | $\varepsilon\in \mathbb{Q_+}$                                                                                                                                                | Abstract                                | Problem hardness parameter                                                                                                                                                                  |
| ε'       | $\varepsilon^\prime = \varepsilon / 5$                                                                                                                                       | First case                              | Less than $\varepsilon$                                                                                                                                                                     |
| fj       | $f_j: B\to\mathbb{R}_{\geq 0}, x \mapsto\sum\limits_{C\in C_I}C(j)\frac{x_C}{p_j}$ for $j\in\mathcal{J}$                                                                     | Lemma 2.1                               | The fraction of job $j$ that is scheuled in a fractional schedule derived from a valid configuration $x$                                                                                    |
| G        | $G=1/\varepsilon^{\prime 2}$                                                                                                                                                 | First case                              | Number of Groups                                                                                                                                                                            |
| I        | $I = (\mathcal{J} , m, R)$                                                                                                                                                   | Scheduling with one additional resource | Problem Instance                                                                                                                                                                            |
| Isup     | $I_{sup} = (\mathcal{J}_{sup} , m, R)$                                                                                                                                       | First case                              | Problem Instance from $\mathcal{J}_{sup}$                                                                                                                                                   |
| j        | $j \in \mathcal{J} $                                                                                                                                                         | Intro                                   | First value of the Job triple. Identifier of the Job                                                                                                                                        |
| J        | $\mathcal{J}=\{(j,p,r)\mid j \in \mathbb{N} ~~ p,r \in\mathbb{Q}$\}                                                                                                          | Scheduling with one additional resource | Set of $n$ Jobs                                                                                                                                                                             |
| JN       | $\mathcal{J}_N=\mathcal{J}\setminus\mathcal{J}_W$                                                                                                                            | First case                              | Set of narrow Jobs                                                                                                                                                                          |
| Jsup     | $\mathcal{J}_{sup} = \mathcal{J_W} \cup \{(n+i, P(\mathcal{J}_{W,i}), R_i)\mid i\in[G-1]\}$                                                                                  | First case                              | Set containing all jobs from $\mathcal{J}_W$ and one additional job for each $i\in\{1,\dots,G-1\} $                                                                                         |
| JsupW    | $J_{sup,W}$                                                                                                                                                                  | First case                              | Widest jobs in $I_{sup}$???                                                                                                                                                                 |
| JW       | $\mathcal{J}_W=\{j\in\mathcal{J\mid r_j\geq\varepsilon^\prime R}\}$                                                                                                          | First case                              | Set of wide Jobs                                                                                                                                                                            |
| JWi      | $\mathcal{J}_{W,i} \subset \mathcal{J}_W$                                                                                                                                    | First case                              | Set of jobs that lie between the lines $(i-1)\varepsilon^{\prime 2}P_W$ and $i\varepsilon^{\prime 2}P_W$ of stacked Jobs sorted by resource amount, called a group                          |
| JWG      | $J_{W,G}$                                                                                                                                                                    | First case                              | Group with the highest resource amount                                                                                                                                                      |
| ki       | $k_i=\lceil P_\text{pre}(K_i)/(\varepsilon^{\prime 2}P_\text{pre})\rceil$ for $0<i\leq 1/\varepsilon^\prime$                                                                 | Lemma 2.3                               | First multiple of $\varepsilon^{\prime 2}P_\text{pre}$ above the stack of generalized configurations, i.e. not intersecting a configuration                                                 |
| Ki       | $K_i=\{C\in\mathcal{C}_W\mid m(C)=i\}$ for $i\in\{1,\dots,m\}$                                                                                                               | Lemma 2.3                               | Set of all configurations using $i$ machines                                                                                                                                                |
| Kik      | $K_{i,k}\subset K_i$                                                                                                                                                         | Lemma 2.3                               | Set of configurations in $K_i$ between $C^{i,k-1}$ and $C^{i,k}$                                                                                                                            |
| lambda*  | $\lambda^\ast=\text{max}\{\lambda\mid \exists x\in B\forall j\leq M:f_j(x)\geq\lambda\}$                                                                                     | Lemma 2.1                               | Largest possible smallest job slice of all valid configurations                                                                                                                             |
| lambda^  | $\hat\lambda=\text{min}\{f_j(x)\mid j\in\mathcal{J}\}$                                                                                                                       | Remark 2.1                              | Fraction how much that job is scheduled which is scheduled the least                                                                                                                        |
| M        | $M=\{f_j\mid j\in\mathcal{J}\}$                                                                                                                                              | Lemma 2.1                               | Set of all $f_j$                                                                                                                                                                            |
| m        | $m \in \mathbb{N}$                                                                                                                                                           | Scheduling with one additional resource | Number of machines                                                                                                                                                                          |
| m(w)     | $m(w)=m_w$                                                                                                                                                                   | Handling the narrow jobs                | Number of machines of a window                                                                                                                                                              |
| µ        | $\mu : \mathcal{J} \to [m]$                                                                                                                                                  | Scheduling with one additional resource | Mapping from jobs to machines machines                                                                                                                                                      |
| n        | $n \in \mathbb{N}$                                                                                                                                                           | Intro                                   | Number of jobs                                                                                                                                                                              |
| pj       | $p_j$                                                                                                                                                                        | Intro                                   | Process time $p$ of job $j$                                                                                                                                                                 |
| Ppre     | $P_\text{pre}=\sum_{C\in\mathcal{C}_\text{pre}}(x_\text{pre})_C$                                                                                                             | Remark 2.1                              | Processing time of the preemptive schedule                                                                                                                                                  |
| PpreC    | $P_\text{pre}(C)=\tilde{x}_{C,w(C)}$                                                                                                                                         | Lemma 2.2                               | Processing time of the generalized configuration in the preemptive schedule                                                                                                                 |
| PpreK    | $P_\text{pre}(K)=\sum_{C\in K}P_\text{pre}(C)=\sum_{C\in K}\tilde{x}_{C,w(C)}$ for $K\subseteq\mathcal{C}_W$                                                                 | Lemma 2.2                               | Total processing time for a set $K$ of configurations of wide jobs                                                                                                                          |
| PW       | $P_W = P(\mathcal{J}_W)$                                                                                                                                                     | First case                              | Processing time of all wide jobs                                                                                                                                                            |
| pmax     | $p_{max} = max\{p_j\mid j \in\mathcal{J}(I)\}$                                                                                                                               | Scheduling with one additional resource | Maximal processing time of all jobs                                                                                                                                                         |
| P(J)     | $P(\mathcal{J}) = \sum_{j\in\mathcal{J}}p_j$                                                                                                                                 | Scheduling with one additional resource | Total processing time of a set of jobs                                                                                                                                                      |
| P(w,x)   | $P(w,x)=\sum\limits_{\substack{C\in\mathcal{C}_W\\C(w)\geq w}}x_{(C,w)}$                                                                                                     | Handling the narrow jobs                | Summed up processing time of a window $w\in\mathcal{W}$ in $x$                                                                                                                              |
| P(x)     | $P(x)=\sum\limits_{C\in\mathcal{C}_W}\sum\limits_{\substack{w\in\mathcal{W}\\w\leq w(C)}}x_{(C,w)}$                                                                          | Handling the narrow jobs                | Makespan of $x$                                                                                                                                                                             |
| phiC     | $\varphi_C=P_\text{pre}(C)/P_\text{pre}(w(C))$                                                                                                                               | Lemma 2.3                               | Amount of processing time to be added to $w(C)$                                                                                                                                             |
| phicar   | $\v{\varphi}_{i,k}=\v{w}_{i,k}/P_\text{pre}(w_{i,k})$                                                                                                                        | Lemma 2.3                               | Fraction of $\v{w}_{i,k}$                                                                                                                                                                   |
| phihat   | $\^{\varphi}_{i,k}=\^{w}_{i,k}/P_\text{pre}(w_{i,k})$                                                                                                                        | Lemma 2.3                               | Fraction of $\^{w}_{i,k}$                                                                                                                                                                   |
| q        | $q=q(\v{x})\in\mathbb{R}^{\|J\|}$                                                                                                                                            | Lemma 2.1                               | Price vaector for the current value $\v{x}$ in Grigoriadis et al.                                                                                                                           |
| rj       | $r_j$                                                                                                                                                                        | Intro                                   | Resource amount $r$ of job $j$                                                                                                                                                              |
| rho      | $\rho=\frac{\varepsilon^\prime}{1+\varepsilon^\prime}\in\mathcal{O}(\varepsilon)$                                                                                            | Lemma 2.1                               | Accuracy of Grigoriadis et al.                                                                                                                                                              |
| R        | $R \in \mathbb{Q}$                                                                                                                                                           | Scheduling with one additional resource | Additional Resource                                                                                                                                                                         |
| R_i      | $R_i = max\{r_j\mid j\in\mathcal{J}_{W,i}\}$                                                                                                                                 | First case                              | Largest resource amount of group $i$. $$\\$$ Note: $R_i \leq R_{i+1}$ for all $i<G$                                                                                                         |
| R(w)     | $R(w)=w_r$                                                                                                                                                                   | Handling the narrow jobs                | Resource amount of a window                                                                                                                                                                 |
| s(Cij)   | $s(C_{i,j})\in\mathbb{Q}$                                                                                                                                                    | Lemma 2.3                               | Start positions for configurations inside $K_i$ with $s(C_{i,1})=0$ and $s(C_{i,j})=s(C_{i,j-1})+P_\text{pre}(C_{i,j-1})$ for $j>1$                                                         |
| t        | $\tau : \mathcal{J} \to \mathbb{Q}$                                                                                                                                          | Scheduling with one additional resource | Mapping from Jobs to starting times                                                                                                                                                         |
| w        | $w=(w_r,w_m)$                                                                                                                                                                | Handling the narrow jobs                | A pair of a resource amount and a number of machines                                                                                                                                        |
| w(C)     | $w(C)=(R-R(C),m-m(C))$                                                                                                                                                       | Handling the narrow jobs                | Main window of a configuration                                                                                                                                                              |
| wcar     | $\v{w}_{i,k}=\varepsilon^{\prime 2}P_\text{pre}\lceil s(C^{i,k})/\varepsilon^{\prime 2}P_\text{pre}\rceil-s(C^{i,k})$                                                        | Lemma 2.3                               | The processing time of window $w_{i,k}$ which has to be scheduled in the window $w_{i,k-1}$                                                                                                 |
| what     | $\^{w}_{i,k}=\tilde{x}_{(C^{i,k},w_{i,k})}-\v{w}_{i,k}$                                                                                                                      | Lemma 2.3                               | The processing time of window $w_{i,k}$ which can stay in window $w_{i,k}$                                                                                                                  |
| W        | $\mathcal{W}$                                                                                                                                                                | Handling the narrow jobs                | Any set of windows                                                                                                                                                                          |
| W'       | $\mathcal{W}^\prime$                                                                                                                                                         | Lemma 2.3                               |                                                                                                                                                                                             |
| wik      | $w_{i,k}=w(C^{i,k})$ and $w_{i,k_i}=(0,0)$                                                                                                                                   | Lemma 2.3                               | Main window of intersected configuration                                                                                                                                                    |
| WKi      | $\mathcal{W}_{K_i}=\{w(C^{i,k})\mid k\in\{1,\dots,k_i-1\}\}$                                                                                                                 | Lemma 2.3                               | Set of chosen windows from $K_i$, it holds that $\mid \mathcal{W}_{K_i}\mid\leq\lfloor P_\text{pre}(K_i)/(\varepsilon^{\prime 2}P_\text{pre})\rfloor$                                       |
| wm       | $w_m$                                                                                                                                                                        | Handling the narrow jobs                | Number of machines inside a window (second component of pair)                                                                                                                               |
| Wpre     | $\mathcal{W}_\text{pre}=\{w(C)\mid C\in\mathcal{C}_{\text{pre},W}\}$                                                                                                         | Handling the narrow jobs                | Set of all main windows in a preemptive schedule                                                                                                                                            |
| wr       | $w_r$                                                                                                                                                                        | Handling the narrow jobs                | Resource amount inside a window (first component of pair)                                                                                                                                   |
| W'       | $\mathcal{W}_\text{pre}\supseteq\mathcal{W}^\prime=\bigcup\limits^{1/\varepsilon^\prime}_{i=1}\mathcal{W}_{K_i}\cup\{(0,0),(R,m)\}$                                          | Lemma 2.3                               | Reduced set of main windows with $\mid\mathcal{W}^\prime\mid\leq2+\varepsilon^{\prime-2}$ corresponding to $(\bar{x},\bar{y})$                                                              |
| w(C)     | if $m(C)=m$ then $(0,0)$ else $w(C)=(R-R(C),m-m(C))$                                                                                                                         | Handling the narrow jobs                | Main window                                                                                                                                                                                 |
| x        | $x\in B$ corresponding to $\lambda^\ast$                                                                                                                                     | Lemma 2.1                               | Valid configuration for which the smallest scheduled job slice is maximal                                                                                                                   |
| xbar     | $\bar{x}_{C,w_{i,k}}=\tilde{x}_{C,w(C)}$ and more                                                                                                                            | Lemma 2.3                               | Solution to $LP_W(\mathcal{W}^\prime)$ for $i\leq 1/e^\prime$, $k\leq k_i$, and $j\in\mathcal{J}_N$                                                                                         |
| xC       | $x_C$                                                                                                                                                                        | Preemptive Schedule                     | Processing time of a valid configuration $C$                                                                                                                                                |
| xcar     | $\v{x}$                                                                                                                                                                      | Lemma 2.1                               | Solution in the current iteration of Grigoriadis et al.                                                                                                                                     |
| xCw      | $x_{(C,w)}=x_C$                                                                                                                                                              | Handling the narrow jobs                | Processing time of the generalized configuration                                                                                                                                            |
| xhat     | $\^{x}$                                                                                                                                                                      | Lemma 2.1                               | Approximative solution of $\text{max}\{q^Tf(x)\mid x\in B\}$                                                                                                                                |
| xpre     | $x_\text{pre}$                                                                                                                                                               | Preemptive Schedule                     | Solution to $LP_\text{pre}(I)$ for a problem instance $I$                                                                                                                                   |
| xpreC    | $(x_\text{pre})_C$                                                                                                                                                           | Remark 2.1                              | Processing time of configuration $C$ in $x_\text{pre}$ (solution to $LP_\text{pre})$                                                                                                        |
| x~       | $\tilde{x}_{C,w(C)}=\sum\limits_{\substack{C^\prime\in\mathcal{C}_\text{pre}\\C^\prime\mid_{J_W}=C}}$ for $C^\prime\in\mathcal{C}_\text{pre}$ reduced to $C\in\mathcal{C}_W$ | Lemma 2.2                               | Processing time of the generalized configuration with a main window                                                                                                                         |
| (x~, y~) | $(\tilde{x},\tilde{y})$                                                                                                                                                      | Lemma 2.2                               | Solution to $LP_W(I,\mathcal{C}_{\text{pre},W},\mathcal{W}_\text{pre})$ which fulfils $P(\tilde{x})=P_\text{pre}$                                                                           |
| ybar     | $\bar{y}_{j,w_{i,k}}=\sum\limits_{C\in K_{i,(k+1)}}\varphi_C\tilde{y}_{j,w(C)}+\^{\varphi}_{i,k}\tilde{y}_{j,w_{i,k}}+\v{\varphi}_{i,k+1}\tilde{y}_{j,w_{i,k+1}}$ and more   | Lemma 2.3                               | Solution to $LP_W(\mathcal{W}^\prime)$ for $i\leq 1/e^\prime$, $k\leq k_i$, and $j\in\mathcal{J}_N$                                                                                         |
| y~       | $\tilde{y}_{j,w}=\sum\limits_{\substack{C\in\mathcal{C}_\text{pre}\\w(C\mid_{\mathcal{J}_W})=w}}C(j)(x_\text{pre})_C$ for window $w\in\mathcal{W}$ and $j\in\mathcal{J}_N$   | Lemma 2.2                               | Sum of the processing times of narrow jobs in configuration window                                                                                                                          |