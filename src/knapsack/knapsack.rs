use crate::knapsack::max_min::KPItem;
use std::{cmp::Ordering, collections::HashMap};

#[derive(Clone)]
struct Store {
    p: f64,
    a: f64,
    tree: Option<Box<StoreTree>>,
}

impl Store {
    fn new(p: f64, a: f64) -> Self {
        Store { p, a, tree: None }
    }

    fn with_predecessor(o: &Store, pj: f64, aj: f64, j: KPItem) -> Self {
        Store {
            p: o.p + pj,
            a: o.a + aj,
            tree: Some(Box::new(StoreTree::new(j, o.tree.clone()))),
        }
    }
}

#[derive(Clone)]
struct StoreTree {
    item: KPItem,
    next: Option<Box<StoreTree>>,
}

impl StoreTree {
    fn new(item: KPItem, next: Option<Box<StoreTree>>) -> Self {
        StoreTree { item, next }
    }
}

fn max_ratio(xs: &Vec<KPItem>) -> Option<&KPItem> {
    xs.into_iter().max_by(|a, b| {
        let ratio_a = a.profit / a.size;
        let ratio_b = b.profit / b.size;
        if ratio_a.partial_cmp(&ratio_b) == Some(Ordering::Equal) {
            a.size.partial_cmp(&b.size).unwrap()
        } else {
            ratio_a.partial_cmp(&ratio_b).unwrap()
        }
    })
}

fn max_profit(xs: &[KPItem]) -> Option<&KPItem> {
    xs.iter()
        .max_by(|a, b| a.profit.partial_cmp(&b.profit).unwrap())
}

fn upper_bound(xs: &Vec<KPItem>, b: i32) -> f64 {
    if let Some(maxr) = max_ratio(xs) {
        let ratio = (b as f64 / maxr.size) as i32;
        if ratio as f64 * maxr.size == b as f64 {
            -1.0
        } else {
            f64::max(ratio as f64 * maxr.profit, max_profit(xs).unwrap().profit)
        }
    } else {
        0.0
    }
}

fn compute_q(large: &[KPItem], t: f64, p: f64) -> Vec<Vec<KPItem>> {
    let up = (p / t).log2().ceil() as usize + 1;
    let mut q = vec![Vec::new(); up];

    for i in large {
        let ind = (i.profit / t).log2().floor() as usize;
        q[ind].push(i.clone());
    }

    q
}

fn reduce_large_items(q: &[Vec<KPItem>]) -> Vec<KPItem> {
    let mut res = Vec::new();
    for qs in q {
        if let Some(max) = qs
            .iter()
            .max_by(|a, b| a.profit.partial_cmp(&b.profit).unwrap())
        {
            res.push(max.clone());
        }
    }
    res
}

fn copy_items(xs: &[KPItem], epsilon: f64, b: i32) -> Vec<KPItem> {
    let mut res = Vec::new();
    let k = (4.0 / epsilon).log2().ceil() as i32;

    for i in xs {
        res.push(i.clone());
        let mut p = 2.0;
        for _ in 0..k {
            if i.size * p <= b as f64 {
                res.push(KPItem::from_item(i, p));
            }
            p *= 2.0;
        }
    }

    res
}

pub fn lawler(xs: Vec<KPItem>, epsilon: f64, b: i32) -> HashMap<KPItem, i32> {
    let p = upper_bound(&xs, b);
    if p == -1.0 {
        let max = max_ratio(&xs).unwrap();
        let ratio = (b as f64 / max.size) as i32;
        let mut res = HashMap::new();
        res.insert(max.clone(), ratio);
        res
    } else {
        let t = (epsilon / 2.0) * p;

        let (small, large) = xs.into_iter().partition::<Vec<_>, _>(|i| i.profit <= t);

        let q = compute_q(&large, t, p);
        let redlarge = reduce_large_items(&q);
        let multredlarge = copy_items(&redlarge, epsilon, b);
        let pairs = produce_pairs(multredlarge, b);

        let mut max;
        let mut phi = 0;
        let mut maxsmall = None;

        if small.is_empty() {
            max = pairs[0].clone();
            let mut profit = 0.0;
            for s in &pairs {
                if s.p > profit {
                    max.clone_from(s);
                    profit = s.p;
                }
            }
        } else {
            maxsmall = max_ratio(&small).cloned();

            max = pairs[0].clone();
            phi = compute_phi(b, maxsmall.as_ref().unwrap(), &max);
            let mut profit = max.p + phi as f64 * maxsmall.as_ref().unwrap().profit;
            for s in &pairs {
                let temp = compute_phi(b, maxsmall.as_ref().unwrap(), s);
                if s.p + temp as f64 * maxsmall.as_ref().unwrap().profit > profit {
                    max.clone_from(s);
                    phi = temp;
                    profit = s.p + temp as f64 * maxsmall.as_ref().unwrap().profit;
                }
            }
        }

        let mut res: HashMap<KPItem, i32> = HashMap::new();
        // let mut res = Pattern::new();
        for i in backtrack(&max) {
            let mult = i.multiplicity as i32;
            res.insert(i, mult);
        }

        if let Some(maxsmall) = maxsmall {
            res.insert(maxsmall, phi);
        }

        res
    }
}

fn produce_pairs(is: Vec<KPItem>, b: i32) -> Vec<Store> {
    let mut list = vec![Store::new(0.0, 0.0)];
    let mut nlist = Vec::new();

    for i in is {
        let pj = i.profit * i.multiplicity as f64;
        let aj = i.size * i.multiplicity as f64;
        for s in &list {
            if s.a + aj <= b as f64 {
                nlist.push(Store::with_predecessor(s, pj, aj, i.clone()));
            }
        }

        list = merge(&list, &nlist);
        nlist.clear();
    }

    list
}

fn compute_phi(b: i32, max: &KPItem, s: &Store) -> i32 {
    let d = b as f64 - s.a;
    (d / max.size) as i32
}

fn merge(xs: &[Store], ys: &[Store]) -> Vec<Store> {
    let mut res: Vec<Store> = Vec::new();
    let mut l = 0;
    let mut r = 0;
    let mut prevp = xs[0].p.min(ys[0].p);

    while l < xs.len() && r < ys.len() {
        let left = &xs[l];
        let right = &ys[r];
        if left.a <= right.a {
            if left.p >= prevp {
                res.push(left.clone());
                prevp = left.p;
            }
            l += 1;
        } else {
            if right.p >= prevp {
                res.push(right.clone());
                prevp = right.p;
            }
            r += 1;
        }
    }

    while l < xs.len() {
        let left = &xs[l];
        if left.p >= prevp {
            res.push(left.clone());
        }
        l += 1;
    }

    while r < ys.len() {
        let right = &ys[r];
        if right.p >= prevp {
            res.push(right.clone());
        }
        r += 1;
    }

    res
}

fn backtrack(opt: &Store) -> Vec<KPItem> {
    let mut act = opt.tree.as_ref();
    let mut res = Vec::new();
    while let Some(tree) = act {
        res.push(tree.item.clone());
        act = tree.next.as_ref();
    }
    res
}

// fn main() {
//     let xs = vec![
//         KPItem::new(1.0, 2.0),
//         KPItem::new(2.0, 3.0),
//         KPItem::new(3.0, 4.0),
//     ];
//     let epsilon = 0.5;
//     let b = 5;
//     let result = lawler(&xs, epsilon, b);
//     println!("{:?}", result);
// }
