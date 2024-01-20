use crate::knapsack::lawler;
use crate::vector::Vector;
use std::hash::{Hash, Hasher};
use std::{cmp::Ordering, collections::HashMap};

#[derive(Clone)]
pub struct BPItem {
    pub id: String,
    pub size: f64,
    pub multiplicity: f64,
}
impl PartialEq for BPItem {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for BPItem {}
// impl Ord for BPItem {
//     fn cmp(&self, other: &Self) -> Ordering {
//         let v1 = self.profit / self.size;
//         let v2 = other.profit / other.size;
//         v1.cmp(v2)
//     }
// }
// impl PartialOrd for BPItem {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         Some(self.cmp(other))
//     }
// }
impl Hash for BPItem {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Pattern(HashMap<BPItem, i32>);
impl Hash for Pattern {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (k, v) in self.0.iter() {
            k.hash(state);
            v.hash(state);
        }
    }
}

#[derive(Clone)]
pub struct KPItem {
    pub id: String,
    pub profit: f64,
    pub size: f64,
    pub multiplicity: f64,
}

impl KPItem {
    pub fn new(id: String, profit: f64, size: f64) -> Self {
        KPItem {
            id,
            profit,
            size,
            multiplicity: 1.0,
        }
    }

    pub fn from_item(item: &KPItem, multiplicity: f64) -> Self {
        KPItem {
            id: item.id.clone(),
            profit: item.profit,
            size: item.size,
            multiplicity,
        }
    }
}

impl PartialEq for KPItem {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for KPItem {}
impl Ord for KPItem {
    fn cmp(&self, other: &Self) -> Ordering {
        let v1 = self.profit / self.size;
        let v2 = other.profit / other.size;
        v1.partial_cmp(&v2).expect("cannot compare KPItems")
    }
}
impl PartialOrd for KPItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Hash for KPItem {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

fn abs_solve(price: &Vec<f64>, t: f64, input: &Vec<BPItem>) -> Vector<Pattern> {
    let mut ass: HashMap<KPItem, BPItem> = HashMap::new();
    let mut items: Vec<KPItem> = Vec::new();

    for (i, item) in input.into_iter().enumerate() {
        let copy = item.clone();
        let id = copy.id;
        let multiplicity = copy.multiplicity;
        let size = copy.size;
        let kp_item = KPItem::new(id, price[i] / multiplicity, size);
        items.push(kp_item.clone());
        ass.insert(kp_item, item.clone());
    }

    let p = lawler(items, t, 1);
    let mut pres = Pattern(HashMap::new());

    for (i, _) in p.iter() {
        if let Some(bp_item) = ass.get(i) {
            pres.0
                .insert(bp_item.clone(), *p.get(i).expect("did not find item"));
        }
    }

    let mut res = Vector::new();
    res.put(pres, 1.0);

    res
}

fn compute_theta_f(t: f64, theta: f64, fx: &[f64]) -> f64 {
    let sum: f64 = fx.iter().map(|&x| theta / (x - theta)).sum();
    sum * (t / fx.len() as f64)
}

fn find_theta(t: f64, fx: &[f64], prec: f64) -> f64 {
    let mut upper = *fx
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
        .unwrap();
    let mut lower = 0.0;
    let mut act = (upper + lower) / 2.0;

    while (act - (upper + lower) / 2.0).abs() > prec {
        act = (upper + lower) / 2.0;
        let val = compute_theta_f(t, act, fx);

        if (val - 1.0).abs() < prec {
            break;
        } else if val - 1.0 < 0.0 {
            lower = act;
        } else {
            upper = act;
        }
    }
    act
}

fn compute_price(fx: &[f64], t: f64, theta: f64) -> Vec<f64> {
    let r = t / fx.len() as f64;
    fx.iter().map(|&x| r * theta / (x - theta)).collect()
}

fn multiply(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

fn apply_all(fs: &Vec<fn(&Vector<Pattern>) -> f64>, x: &Vector<Pattern>) -> Vec<f64> {
    fs.iter().map(|f| f(x)).collect()
}

fn line_search(fx: &[f64], fy: &[f64], theta: f64, t: f64, epsilon: f64) -> f64 {
    let mut up = 1.0;
    let mut low = 0.0;

    while low < (1.0 - epsilon) * up {
        let act = (up + low) / 2.0;
        let defined = fx
            .iter()
            .zip(fy.iter())
            .all(|(&fxi, &fyi)| fxi + act * (fyi - fxi) > theta);

        if !defined {
            up = act;
        } else {
            let val = derivative_pot(act, fx, fy, t, theta);
            if val > 0.0 {
                low = act;
            } else {
                up = act;
            }
        }
    }
    (low + up) / 2.0
}

fn derivative_pot(tau: f64, fx: &[f64], fy: &[f64], t: f64, theta: f64) -> f64 {
    let res: f64 = fx
        .iter()
        .zip(fy.iter())
        .map(|(&fxi, &fyi)| (fyi - fxi) / (fxi + tau * (fyi - fxi) - theta))
        .sum();
    res * t / fx.len() as f64
}

fn compute_v(p: &[f64], fx: &[f64], fy: &[f64]) -> f64 {
    let a = multiply(p, fy);
    let b = multiply(p, fx);
    (a - b) / (b + a)
}

// fn compute_tau(t: f64, theta: f64, v: f64, price: &[f64], fx: &[f64], fy: &[f64]) -> f64 {
//     let a = t * theta * v;
//     let b = 2.0 * fx.len() as f64 * (multiply(price, fx) + multiply(price, fy));
//     a / b
// }

fn compute_start(input: &Vec<BPItem>, m: usize) -> Vector<Pattern> {
    let mut res = Vector::new();
    for i in 0..m {
        let u = unit(i, m);
        let abs_result = abs_solve(&u, 0.5, input);
        res = res.add(&abs_result).scale(1.0 / (m as f64));
    }
    // for (_, value) in res.iter_mut() {
    //     *value /= m as f64;
    // }
    res
}

fn unit(ind: usize, m: usize) -> Vec<f64> {
    (0..m).map(|i| if i == ind { 1.0 } else { 0.0 }).collect()
}

pub fn maxmin(
    input: Vec<BPItem>,
    fs: &Vec<fn(&Vector<Pattern>) -> f64>,
    epsilon: f64,
) -> Vector<Pattern> {
    let mut x = compute_start(&input, fs.len());
    let mut act_epsilon = 0.25;
    let prec = epsilon * epsilon / fs.len() as f64;
    let mut fx = apply_all(fs, &x);
    let mut price: Vec<f64>;

    while act_epsilon > epsilon {
        act_epsilon /= 2.0;
        let t = epsilon / 6.0;
        let min_x = *fx
            .iter()
            .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
            .unwrap();

        while {
            let theta = find_theta(t, &fx, prec);
            price = compute_price(&fx, t, theta);
            let y = abs_solve(&price, t, &input);
            let fy = apply_all(fs, &y);
            let min_y = *fy
                .iter()
                .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                .unwrap();
            let v = compute_v(&price, &fx, &fy);

            if v <= t || min_y >= min_x * (1.0 - act_epsilon) / (1.0 - 2.0 * act_epsilon) {
                false
            } else {
                let tau = line_search(&fx, &fy, theta, t, epsilon);
                let fx1: Vec<f64> = fx
                    .iter()
                    .zip(fy.iter())
                    .map(|(&fxi, &fyi)| (1.0 - tau) * fxi + tau * fyi)
                    .collect();
                x = x.scale(1.0 - tau).add(&y.scale(tau));
                fx = fx1;
                true
            }
        } {}
    }

    x
}
