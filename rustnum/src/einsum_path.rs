//! `einsum_path`: contraction order and report for an einsum expression (port of NumPy's algorithm).
use crate::{Array, Element};
use std::collections::{BTreeMap, BTreeSet};

type Set = BTreeSet<char>;

fn size_of(indices: impl IntoIterator<Item = char>, dims: &BTreeMap<char, usize>) -> u128 {
    indices.into_iter().map(|c| dims[&c] as u128).product()
}
fn flop_count(contract: &Set, inner: bool, num_terms: usize, dims: &BTreeMap<char, usize>) -> u128 {
    let mut factor = num_terms.saturating_sub(1).max(1) as u128;
    if inner {
        factor += 1;
    }
    size_of(contract.iter().copied(), dims) * factor
}
/// Returns (new_result, remaining sets with new_result appended, removed, contracted).
fn find_contraction(
    positions: &[usize],
    input_sets: &[Set],
    output: &Set,
) -> (Set, Vec<Set>, Set, Set) {
    let mut contract = Set::new();
    let mut remain = output.clone();
    let mut remaining = Vec::new();
    for (i, s) in input_sets.iter().enumerate() {
        if positions.contains(&i) {
            contract.extend(s.iter().copied());
        } else {
            remaining.push(s.clone());
            remain.extend(s.iter().copied());
        }
    }
    let new_result: Set = remain.intersection(&contract).copied().collect();
    let removed: Set = contract.difference(&new_result).copied().collect();
    remaining.push(new_result.clone());
    (new_result, remaining, removed, contract)
}

#[derive(Clone)]
struct Candidate {
    sort: (i128, u128),
    pos: (usize, usize),
    sets: Vec<Set>,
}
fn possible_contraction(
    pos: (usize, usize),
    input_sets: &[Set],
    output: &Set,
    dims: &BTreeMap<char, usize>,
    memory_limit: u128,
    path_cost: u128,
    naive_cost: u128,
) -> Option<Candidate> {
    let (result, sets, removed, contract) = find_contraction(&[pos.0, pos.1], input_sets, output);
    let new_size = size_of(result.iter().copied(), dims);
    if new_size > memory_limit {
        return None;
    }
    let old: u128 = [pos.0, pos.1]
        .iter()
        .map(|&p| size_of(input_sets[p].iter().copied(), dims))
        .sum();
    let removed_size = old as i128 - new_size as i128;
    let cost = flop_count(&contract, !removed.is_empty(), 2, dims);
    if path_cost + cost > naive_cost {
        return None;
    }
    Some(Candidate {
        sort: (-removed_size, cost),
        pos,
        sets,
    })
}
fn update_other(results: Vec<Candidate>, best: &Candidate) -> Vec<Candidate> {
    let (bx, by) = (best.pos.0 as isize, best.pos.1 as isize);
    let mut out = Vec::new();
    for mut c in results {
        let (x, y) = (c.pos.0 as isize, c.pos.1 as isize);
        if x == bx || x == by || y == bx || y == by {
            continue;
        }
        let i1 = by - (by > x) as isize - (by > y) as isize;
        c.sets.remove(i1 as usize);
        let i2 = bx - (bx > x) as isize - (bx > y) as isize;
        c.sets.remove(i2 as usize);
        let last = best.sets.last().expect("non-empty").clone();
        let at = c.sets.len().saturating_sub(1);
        c.sets.insert(at, last);
        c.pos = (
            (x - (x > bx) as isize - (x > by) as isize) as usize,
            (y - (y > bx) as isize - (y > by) as isize) as usize,
        );
        out.push(c);
    }
    out
}
fn greedy_path(
    input_sets: &[Set],
    output: &Set,
    dims: &BTreeMap<char, usize>,
    memory_limit: u128,
) -> Vec<Vec<usize>> {
    if input_sets.len() == 1 {
        return vec![vec![0]];
    }
    if input_sets.len() == 2 {
        return vec![vec![0, 1]];
    }
    let all: Vec<usize> = (0..input_sets.len()).collect();
    let (_, _, removed, contract) = find_contraction(&all, input_sets, output);
    let naive_cost = flop_count(&contract, !removed.is_empty(), input_sets.len(), dims);
    let mut sets: Vec<Set> = input_sets.to_vec();
    let mut pairs: Vec<(usize, usize)> = (0..sets.len())
        .flat_map(|i| ((i + 1)..sets.len()).map(move |j| (i, j)))
        .collect();
    let mut known: Vec<Candidate> = Vec::new();
    let mut path_cost = 0u128;
    let mut path: Vec<Vec<usize>> = Vec::new();
    for _ in 0..input_sets.len() - 1 {
        for &p in &pairs {
            if sets[p.0].is_disjoint(&sets[p.1]) {
                continue;
            }
            if let Some(c) =
                possible_contraction(p, &sets, output, dims, memory_limit, path_cost, naive_cost)
            {
                known.push(c);
            }
        }
        if known.is_empty() {
            for i in 0..sets.len() {
                for j in (i + 1)..sets.len() {
                    if let Some(c) = possible_contraction(
                        (i, j),
                        &sets,
                        output,
                        dims,
                        memory_limit,
                        path_cost,
                        naive_cost,
                    ) {
                        known.push(c);
                    }
                }
            }
            if known.is_empty() {
                path.push((0..sets.len()).collect());
                break;
            }
        }
        let mut best_i = 0;
        for (i, c) in known.iter().enumerate() {
            if c.sort < known[best_i].sort {
                best_i = i;
            }
        }
        let best = known[best_i].clone();
        known = update_other(known, &best);
        sets = best.sets.clone();
        let new_pos = sets.len() - 1;
        pairs = (0..new_pos).map(|i| (i, new_pos)).collect();
        path.push(vec![best.pos.0, best.pos.1]);
        path_cost += best.sort.1;
    }
    path
}
fn optimal_path(
    input_sets: &[Set],
    output: &Set,
    dims: &BTreeMap<char, usize>,
    memory_limit: u128,
) -> Vec<Vec<usize>> {
    type Entry = (u128, Vec<Vec<usize>>, Vec<Set>);
    let mut full: Vec<Entry> = vec![(0, vec![], input_sets.to_vec())];
    for iteration in 0..input_sets.len() - 1 {
        let mut iter_results: Vec<Entry> = Vec::new();
        for (cost, positions, remaining) in &full {
            let n = input_sets.len() - iteration;
            for i in 0..n {
                for j in (i + 1)..n {
                    let (result, sets, removed, contract) =
                        find_contraction(&[i, j], remaining, output);
                    if size_of(result.iter().copied(), dims) > memory_limit {
                        continue;
                    }
                    let total = cost + flop_count(&contract, !removed.is_empty(), 2, dims);
                    let mut p = positions.clone();
                    p.push(vec![i, j]);
                    iter_results.push((total, p, sets));
                }
            }
        }
        if !iter_results.is_empty() {
            full = iter_results;
        } else {
            let mut best = 0;
            for (i, e) in full.iter().enumerate() {
                if e.0 < full[best].0 {
                    best = i;
                }
            }
            let mut path = full[best].1.clone();
            path.push((0..input_sets.len() - iteration).collect());
            return path;
        }
    }
    if full.is_empty() {
        return vec![(0..input_sets.len()).collect()];
    }
    let mut best = 0;
    for (i, e) in full.iter().enumerate() {
        if e.0 < full[best].0 {
            best = i;
        }
    }
    full[best].1.clone()
}

fn parse_input(subscripts: &str, ndims: &[usize]) -> (Vec<String>, String) {
    let mut subs: String = subscripts.chars().filter(|c| *c != ' ').collect();
    for c in subs.chars() {
        assert!(
            c.is_ascii_alphabetic() || ".,->".contains(c),
            "Character {c} is not a valid symbol."
        );
    }
    if subs.contains('-') || subs.contains('>') {
        let invalid = subs.matches('-').count() > 1 || subs.matches('>').count() > 1;
        assert!(
            !invalid && subs.matches("->").count() == 1,
            "Subscripts can only contain one '->'."
        );
    }
    if subs.contains('.') {
        let used: Set = subs.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        let unused: Vec<char> = ('A'..='Z')
            .chain('a'..='z')
            .filter(|c| !used.contains(c))
            .collect();
        let (input_part, output_part) = match subs.split_once("->") {
            Some((i, o)) => (i.to_string(), Some(o.to_string())),
            None => (subs.clone(), None),
        };
        let mut terms: Vec<String> = input_part.split(',').map(str::to_string).collect();
        let mut longest = 0usize;
        for (num, sub) in terms.iter_mut().enumerate() {
            if sub.contains('.') {
                assert!(
                    sub.matches('.').count() == 3 && sub.matches("...").count() == 1,
                    "Invalid Ellipses."
                );
                let count = if ndims[num] == 0 {
                    0
                } else {
                    ndims[num] as isize - (sub.len() as isize - 3)
                };
                assert!(count >= 0, "Ellipses lengths do not match.");
                let count = count as usize;
                longest = longest.max(count);
                let rep: String = unused[unused.len() - count..].iter().collect();
                *sub = sub.replace("...", &rep);
            }
        }
        let out_ellipse: String = unused[unused.len() - longest..].iter().collect();
        subs = terms.join(",");
        match output_part {
            Some(o) => subs = format!("{subs}->{}", o.replace("...", &out_ellipse)),
            None => {
                let flat: String = subs.replace(',', "");
                let once: String = flat
                    .chars()
                    .collect::<Set>()
                    .into_iter()
                    .filter(|c| flat.matches(*c).count() == 1)
                    .collect();
                let normal: String = once.chars().filter(|c| !out_ellipse.contains(*c)).collect();
                subs = format!("{subs}->{out_ellipse}{normal}");
            }
        }
    }
    let (inputs, output) = match subs.split_once("->") {
        Some((i, o)) => (i.to_string(), o.to_string()),
        None => {
            let flat: String = subs.replace(',', "");
            let once: String = flat
                .chars()
                .collect::<Set>()
                .into_iter()
                .filter(|c| flat.matches(*c).count() == 1)
                .collect();
            (subs.clone(), once)
        }
    };
    for c in output.chars() {
        assert!(
            output.matches(c).count() == 1,
            "Output character {c} appeared more than once in the output."
        );
        assert!(
            inputs.contains(c),
            "Output character {c} did not appear in the input"
        );
    }
    let terms: Vec<String> = inputs.split(',').map(str::to_string).collect();
    assert!(
        terms.len() == ndims.len(),
        "Number of einsum subscripts must be equal to the number of operands."
    );
    (terms, output)
}
fn sci(x: f64) -> String {
    let s = format!("{x:.3e}");
    let (m, e) = s.split_once('e').expect("exponent");
    let e: i32 = e.parse().expect("exponent digits");
    format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
}

/// Lowest-cost contraction order of an einsum expression plus a report, like np.einsum_path(subscripts, *operands, optimize).
///
/// `optimize` is "greedy" (NumPy's default), "optimal" (exhaustive, factorial cost) or "false"
/// (no optimisation: one step contracting everything). The path lists, per step, the positions of
/// the operands to contract; each result is appended to the end of the operand list. The report is
/// NumPy's text, including the naive and optimised FLOP counts and the per-step table. With an
/// ellipsis, NumPy picks unused index letters at random; rustnum picks the last unused letters in
/// alphabetical order.
/// ```
/// use rustnum as np;
/// let a = np::zeros::<f64>([2, 2]);
/// let b = np::zeros::<f64>([2, 5]);
/// let c = np::zeros::<f64>([5, 2]);
/// let (path, report) = np::einsum_path("ij,jk,kl->il", &[&a, &b, &c], "greedy");
/// assert_eq!(path, [vec![1, 2], vec![0, 1]]);
/// assert!(report.contains("Optimized scaling:  3"));
/// ```
pub fn einsum_path<T: Element>(
    subscripts: &str,
    operands: &[&Array<T>],
    optimize: &str,
) -> (Vec<Vec<usize>>, String) {
    let ndims: Vec<usize> = operands.iter().map(|a| a.ndim()).collect();
    let (input_list_orig, output_sub) = parse_input(subscripts, &ndims);
    let mut input_list = input_list_orig.clone();
    let mut input_sets: Vec<Set> = input_list.iter().map(|s| s.chars().collect()).collect();
    let output_set: Set = output_sub.chars().collect();
    let indices: Set = input_list.iter().flat_map(|s| s.chars()).collect();
    let mut dims: BTreeMap<char, usize> = BTreeMap::new();
    for (t, term) in input_list.iter().enumerate() {
        let sh = operands[t].shape();
        assert!(
            sh.len() == term.chars().count(),
            "Einstein sum subscript {term} does not contain the correct number of indices for operand {t}."
        );
        for (c, ch) in term.chars().enumerate() {
            let dim = sh[c];
            match dims.get(&ch).copied() {
                Some(1) => {
                    dims.insert(ch, dim);
                }
                Some(prev) => assert!(
                    dim == 1 || dim == prev,
                    "Size of label '{ch}' for operand {t} ({prev}) does not match previous terms ({dim})."
                ),
                None => {
                    dims.insert(ch, dim);
                }
            }
        }
    }
    let max_size = input_list
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(output_sub.as_str()))
        .map(|t| size_of(t.chars(), &dims))
        .max()
        .unwrap_or(1);
    let total_chars: usize = input_sets.iter().map(|s| s.len()).sum();
    let inner_product = total_chars > indices.len();
    let naive_cost = flop_count(&indices, inner_product, input_list.len(), &dims);
    let path: Vec<Vec<usize>> = if optimize == "false"
        || input_list.len() == 1
        || input_list.len() == 2
        || indices == output_set
    {
        vec![(0..input_list.len()).collect()]
    } else if optimize == "greedy" || optimize == "true" {
        greedy_path(&input_sets, &output_set, &dims, max_size)
    } else if optimize == "optimal" {
        optimal_path(&input_sets, &output_set, &dims, max_size)
    } else {
        panic!("Path name {optimize} not found")
    };
    let (mut cost_list, mut scale_list, mut size_list) = (Vec::new(), Vec::new(), Vec::new());
    let mut contractions: Vec<(String, Vec<String>)> = Vec::new();
    for (cnum, step) in path.iter().enumerate() {
        let mut inds = step.clone();
        inds.sort_unstable_by(|a, b| b.cmp(a));
        let (out_inds, rest, removed, contract) = find_contraction(&inds, &input_sets, &output_set);
        input_sets = rest;
        cost_list.push(flop_count(
            &contract,
            !removed.is_empty(),
            inds.len(),
            &dims,
        ));
        scale_list.push(contract.len());
        size_list.push(size_of(out_inds.iter().copied(), &dims));
        let tmp: Vec<String> = inds.iter().map(|&x| input_list.remove(x)).collect();
        let result: String = if cnum + 1 == path.len() {
            output_sub.clone()
        } else {
            let mut v: Vec<(usize, char)> = out_inds.iter().map(|&c| (dims[&c], c)).collect();
            v.sort();
            v.into_iter().map(|x| x.1).collect()
        };
        input_list.push(result.clone());
        contractions.push((format!("{}->{}", tmp.join(","), result), input_list.clone()));
    }
    let opt_cost: u128 = cost_list.iter().sum::<u128>() + 1;
    assert!(
        input_list.len() == 1,
        "Invalid einsum_path is specified: {} more operands has to be contracted.",
        input_list.len() - 1
    );
    let speedup = naive_cost as f64 / opt_cost as f64;
    let max_i = size_list.iter().copied().max().unwrap_or(0);
    let mut out = format!(
        "  Complete contraction:  {}->{}\n",
        input_list_orig.join(","),
        output_sub
    );
    out += &format!("         Naive scaling:  {}\n", indices.len());
    out += &format!(
        "     Optimized scaling:  {}\n",
        scale_list.iter().max().copied().unwrap_or(0)
    );
    out += &format!("      Naive FLOP count:  {}\n", sci(naive_cost as f64));
    out += &format!("  Optimized FLOP count:  {}\n", sci(opt_cost as f64));
    out += &format!("   Theoretical speedup:  {speedup:3.3}\n");
    out += &format!("  Largest intermediate:  {} elements\n", sci(max_i as f64));
    out += &"-".repeat(74);
    out += "\n";
    out += &format!("{:>6} {:>24} {:>40}\n", "scaling", "current", "remaining");
    out += &"-".repeat(74);
    for (n, (einsum_str, remaining)) in contractions.iter().enumerate() {
        let remaining_str = format!("{}->{}", remaining.join(","), output_sub);
        out += &format!(
            "\n{:>4}    {:>24} {:>40}",
            scale_list[n], einsum_str, remaining_str
        );
    }
    (path, out)
}
