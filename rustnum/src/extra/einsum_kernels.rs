//! `einsum` kernel: subscript parsing plus a general strided nested-loop contraction.
//!
//! Supported: any number of operands, explicit (`"ij,jk->ik"`) and implicit (`"ij,jk"`)
//! output, repeated labels inside one operand (diagonal / trace), summed labels, broadcasting
//! of size-1 dimensions, spaces, and `...` ellipsis (right-aligned and broadcast, as in
//! NumPy). Not supported: `optimize=`, `out=`, the interleaved `einsum(op, [sublist], ...)` form.
//!
//! The contraction visits the output indices in C order and, for each, sums the product of the
//! operands over all summed labels. No intermediate arrays, so memory is O(output). Cost is the
//! product of every label's size times the number of operands, i.e. no pairwise contraction
//! ordering is attempted ("correctness first").

use crate::Num;
use crate::element::{add, mul};
use std::collections::BTreeMap;

/// A strided view over a slice: element `index` is `data[offset + sum(index[d] * strides[d])]`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Operand<'a, T> {
    pub data: &'a [T],
    pub shape: &'a [usize],
    pub strides: &'a [isize],
    pub offset: usize,
}

/// C-order strides (in elements) for `shape`.
pub(crate) fn c_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut acc = 1isize;
    for (s, &d) in strides.iter_mut().zip(shape).rev() {
        *s = acc;
        acc *= d.max(1) as isize;
    }
    strides
}

/// A label is a letter (its ASCII code) or an ellipsis axis (`ELLIPSIS + position`).
type Label = u32;
const ELLIPSIS: Label = 256;

/// Subscripts for one operand before ellipsis resolution.
#[derive(Debug, PartialEq)]
struct RawTerm {
    labels: Vec<Label>,
    /// Position of `...` among `labels` (the ellipsis expands there), if present.
    ellipsis_at: Option<usize>,
}

fn parse_term(term: &str, allow_ellipsis: bool) -> RawTerm {
    let chars: Vec<char> = term.chars().filter(|c| !c.is_whitespace()).collect();
    let mut labels = Vec::new();
    let mut ellipsis_at = None;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '.' {
            if !allow_ellipsis
                || ellipsis_at.is_some()
                || chars.get(i..i + 3) != Some(&['.', '.', '.'])
            {
                panic!(
                    "einstein sum subscripts string contains a '.' that is not part of an ellipsis ('...')"
                );
            }
            ellipsis_at = Some(labels.len());
            i += 3;
        } else if c.is_ascii_alphabetic() {
            labels.push(c as Label);
            i += 1;
        } else {
            panic!(
                "invalid subscript '{c}' in einstein sum subscripts string, subscripts must be letters"
            );
        }
    }
    RawTerm {
        labels,
        ellipsis_at,
    }
}

/// Expand a term's ellipsis for an operand of `ndim` dimensions; ellipsis dimensions are
/// right-aligned inside the global ellipsis block of `max_ell` axes.
fn resolve(term: &RawTerm, ndim: usize, max_ell: usize, op: usize) -> Vec<Label> {
    let explicit = term.labels.len();
    match term.ellipsis_at {
        None => {
            if ndim != explicit {
                if ndim > explicit {
                    panic!(
                        "operand has more dimensions than subscripts given in einstein sum, but no '...' ellipsis provided to broadcast the extra dimensions."
                    );
                }
                panic!(
                    "einstein sum subscripts string contains too many subscripts for operand {op}"
                );
            }
            term.labels.clone()
        }
        Some(at) => {
            if ndim < explicit {
                panic!(
                    "einstein sum subscripts string contains too many subscripts for operand {op}"
                );
            }
            let cnt = ndim - explicit;
            let mut out = term.labels[..at].to_vec();
            out.extend((0..cnt).map(|j| ELLIPSIS + (max_ell - cnt + j) as Label));
            out.extend_from_slice(&term.labels[at..]);
            out
        }
    }
}

/// Parse `subscripts` for operands with the given shapes: returns each operand's resolved
/// labels (one per dimension) and the output labels.
fn parse(subscripts: &str, ndims: &[usize]) -> (Vec<Vec<Label>>, Vec<Label>) {
    let (lhs, rhs) = match subscripts.split_once("->") {
        Some((l, r)) => (l, Some(r)),
        None => (subscripts, None),
    };
    let terms: Vec<RawTerm> = lhs.split(',').map(|t| parse_term(t, true)).collect();
    if terms.len() > ndims.len() {
        panic!("more operands in einstein sum subscripts string than were provided");
    }
    if terms.len() < ndims.len() {
        panic!(
            "more operands provided to einstein sum function than specified in the subscripts string"
        );
    }
    let max_ell = terms
        .iter()
        .zip(ndims)
        .filter(|(t, _)| t.ellipsis_at.is_some())
        .map(|(t, &n)| n.saturating_sub(t.labels.len()))
        .max()
        .unwrap_or(0);
    let inputs: Vec<Vec<Label>> = terms
        .iter()
        .zip(ndims)
        .enumerate()
        .map(|(k, (t, &n))| resolve(t, n, max_ell, k))
        .collect();

    let output = match rhs {
        Some(r) => {
            let term = parse_term(r, true);
            let mut out = term.labels.clone();
            if let Some(at) = term.ellipsis_at {
                let block = (0..max_ell).map(|j| ELLIPSIS + j as Label);
                out.splice(at..at, block);
            } else if max_ell > 0 {
                panic!(
                    "output has more dimensions than subscripts given in einstein sum, but no '...' ellipsis provided to broadcast the extra dimensions."
                );
            }
            for (i, l) in out.iter().enumerate() {
                if out[..i].contains(l) {
                    panic!(
                        "einstein sum subscripts string includes output subscript '{}' multiple times",
                        char::from(*l as u8)
                    );
                }
                if !inputs.iter().any(|t| t.contains(l)) {
                    panic!(
                        "einstein sum subscripts string included output subscript '{}' which never appeared in an input",
                        char::from(*l as u8)
                    );
                }
            }
            out
        }
        None => {
            // Implicit mode: ellipsis axes first, then letters that occur exactly once, sorted.
            let mut count: BTreeMap<Label, usize> = BTreeMap::new();
            for l in inputs.iter().flatten() {
                *count.entry(*l).or_insert(0) += 1;
            }
            let mut out: Vec<Label> = (0..max_ell).map(|j| ELLIPSIS + j as Label).collect();
            out.extend(
                count
                    .iter()
                    .filter(|(l, c)| **l < ELLIPSIS && **c == 1)
                    .map(|(l, _)| *l),
            );
            out
        }
    };
    (inputs, output)
}

/// `np.einsum(subscripts, *operands)` over strided operands.
///
/// Returns the C-ordered result and its shape (empty shape = scalar, one element).
pub(crate) fn einsum<T>(subscripts: &str, operands: &[Operand<'_, T>]) -> (Vec<T>, Vec<usize>)
where
    T: Num,
{
    if operands.is_empty() {
        panic!("must provide at least an operand");
    }
    let ndims: Vec<usize> = operands.iter().map(|o| o.shape.len()).collect();
    let (inputs, output) = parse(subscripts, &ndims);

    // Size of every label; a size-1 dimension broadcasts against anything.
    let mut size: BTreeMap<Label, usize> = BTreeMap::new();
    for (k, (op, labels)) in operands.iter().zip(&inputs).enumerate() {
        for (a, &l) in labels.iter().enumerate() {
            let d = op.shape[a];
            if let Some(b) = labels[..a].iter().position(|m| *m == l)
                && op.shape[b] != d
            {
                panic!(
                    "dimensions in operand {k} for collapsing index '{}' don't match ({} != {})",
                    char::from(l as u8),
                    op.shape[b],
                    d
                );
            }
            let e = size.entry(l).or_insert(1);
            if d != 1 {
                if *e != 1 && *e != d {
                    panic!(
                        "operands could not be broadcast together with remapped shapes [original->remapped]"
                    );
                }
                *e = d;
            }
        }
    }

    // Loop axes: output labels first, then summed labels in sorted order.
    let mut loop_labels = output.clone();
    loop_labels.extend(size.keys().filter(|l| !output.contains(l)));
    let sizes: Vec<usize> = loop_labels.iter().map(|l| size[l]).collect();
    let n_out = output.len();

    // Per operand, per loop axis: total stride (repeated labels add up, giving the diagonal).
    let op_strides: Vec<Vec<isize>> = operands
        .iter()
        .zip(&inputs)
        .map(|(op, labels)| {
            loop_labels
                .iter()
                .zip(&sizes)
                .map(|(l, &sz)| {
                    labels
                        .iter()
                        .enumerate()
                        .filter(|(_, m)| *m == l)
                        .map(|(a, _)| {
                            if op.shape[a] == 1 && sz > 1 {
                                0
                            } else {
                                op.strides[a]
                            }
                        })
                        .sum()
                })
                .collect()
        })
        .collect();

    let out_shape: Vec<usize> = sizes[..n_out].to_vec();
    let out_len: usize = out_shape.iter().product();
    let red_sizes = &sizes[n_out..];
    let red_len: usize = red_sizes.iter().product();
    let nops = operands.len();

    let mut out = Vec::with_capacity(out_len);
    let mut out_idx = vec![0usize; n_out];
    let mut base: Vec<isize> = operands.iter().map(|o| o.offset as isize).collect();
    for _ in 0..out_len {
        let mut acc = T::default();
        let mut offs = base.clone();
        let mut red_idx = vec![0usize; red_sizes.len()];
        for _ in 0..red_len {
            let mut p = operands[0].data[offs[0] as usize];
            for k in 1..nops {
                p = mul(p, operands[k].data[offs[k] as usize]);
            }
            acc = add(acc, p);
            for ax in (0..red_sizes.len()).rev() {
                red_idx[ax] += 1;
                for k in 0..nops {
                    offs[k] += op_strides[k][n_out + ax];
                }
                if red_idx[ax] < red_sizes[ax] {
                    break;
                }
                for k in 0..nops {
                    offs[k] -= op_strides[k][n_out + ax] * red_sizes[ax] as isize;
                }
                red_idx[ax] = 0;
            }
        }
        out.push(acc);
        for ax in (0..n_out).rev() {
            out_idx[ax] += 1;
            for k in 0..nops {
                base[k] += op_strides[k][ax];
            }
            if out_idx[ax] < out_shape[ax] {
                break;
            }
            for k in 0..nops {
                base[k] -= op_strides[k][ax] * out_shape[ax] as isize;
            }
            out_idx[ax] = 0;
        }
    }
    (out, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::assert_close;

    fn check(sub: &str, ops: &[(&[f64], &[usize])], want: &[f64], want_shape: &[usize]) {
        let strides: Vec<Vec<isize>> = ops.iter().map(|(_, s)| c_strides(s)).collect();
        let operands: Vec<Operand<f64>> = ops
            .iter()
            .zip(&strides)
            .map(|((d, s), st)| Operand {
                data: d,
                shape: s,
                strides: st,
                offset: 0,
            })
            .collect();
        let (got, shape) = einsum(sub, &operands);
        assert_eq!(shape, want_shape, "shape of {sub}");
        assert_close(&got, want);
    }

    #[test]
    fn einsum_matches_numpy() {
        check(
            "ij,jk->ik",
            &[
                (
                    &[
                        -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                    ],
                    &[3, 4],
                ),
                (
                    &[
                        0.0, -4.0, 0.0, -3.0, 2.0, 4.0, 4.0, 1.0, 3.0, -1.0, -3.0, 0.0, -1.0, 1.0,
                        4.0, -2.0, 3.0, -3.0, -1.0, 3.0,
                    ],
                    &[4, 5],
                ),
            ],
            &[
                -21.0, 0.0, -6.0, 3.0, 9.0, 6.0, -12.0, 11.0, 6.0, -3.0, -17.0, 0.0, -14.0, -14.0,
                11.0,
            ],
            &[3, 5],
        );
        check(
            "ij,jk",
            &[
                (
                    &[
                        -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                    ],
                    &[3, 4],
                ),
                (
                    &[
                        0.0, -4.0, 0.0, -3.0, 2.0, 4.0, 4.0, 1.0, 3.0, -1.0, -3.0, 0.0, -1.0, 1.0,
                        4.0, -2.0, 3.0, -3.0, -1.0, 3.0,
                    ],
                    &[4, 5],
                ),
            ],
            &[
                -21.0, 0.0, -6.0, 3.0, 9.0, 6.0, -12.0, 11.0, 6.0, -3.0, -17.0, 0.0, -14.0, -14.0,
                11.0,
            ],
            &[3, 5],
        );
        check(
            "ij,jk,kl->il",
            &[
                (
                    &[
                        -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                    ],
                    &[3, 4],
                ),
                (
                    &[
                        0.0, -4.0, 0.0, -3.0, 2.0, 4.0, 4.0, 1.0, 3.0, -1.0, -3.0, 0.0, -1.0, 1.0,
                        4.0, -2.0, 3.0, -3.0, -1.0, 3.0,
                    ],
                    &[4, 5],
                ),
                (
                    &[-2.0, 2.0, 0.0, 0.0, 4.0, 3.0, 3.0, 0.0, 4.0, 4.0],
                    &[5, 2],
                ),
            ],
            &[63.0, -24.0, 38.0, 33.0, -20.0, -32.0],
            &[3, 2],
        );
        check(
            "ab,bc,cd",
            &[
                (
                    &[
                        -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                    ],
                    &[3, 4],
                ),
                (
                    &[
                        0.0, -4.0, 0.0, -3.0, 2.0, 4.0, 4.0, 1.0, 3.0, -1.0, -3.0, 0.0, -1.0, 1.0,
                        4.0, -2.0, 3.0, -3.0, -1.0, 3.0,
                    ],
                    &[4, 5],
                ),
                (
                    &[-2.0, 2.0, 0.0, 0.0, 4.0, 3.0, 3.0, 0.0, 4.0, 4.0],
                    &[5, 2],
                ),
            ],
            &[63.0, -24.0, 38.0, 33.0, -20.0, -32.0],
            &[3, 2],
        );
        check(
            "ii->i",
            &[(&[-3.0, -3.0, -2.0, 0.0, 3.0, 0.0, 4.0, -1.0, 4.0], &[3, 3])],
            &[-3.0, 3.0, 4.0],
            &[3],
        );
        check(
            "ii",
            &[(&[-3.0, -3.0, -2.0, 0.0, 3.0, 0.0, 4.0, -1.0, 4.0], &[3, 3])],
            &[4.0],
            &[],
        );
        check(
            "ii->",
            &[(&[-3.0, -3.0, -2.0, 0.0, 3.0, 0.0, 4.0, -1.0, 4.0], &[3, 3])],
            &[4.0],
            &[],
        );
        check(
            "ij->ji",
            &[(
                &[
                    -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                ],
                &[3, 4],
            )],
            &[
                -3.0, 1.0, 0.0, -3.0, 1.0, -3.0, 3.0, 2.0, -1.0, 0.0, -4.0, 4.0,
            ],
            &[4, 3],
        );
        check(
            "ji",
            &[(
                &[
                    -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                ],
                &[3, 4],
            )],
            &[
                -3.0, 1.0, 0.0, -3.0, 1.0, -3.0, 3.0, 2.0, -1.0, 0.0, -4.0, 4.0,
            ],
            &[4, 3],
        );
        check(
            "ij",
            &[(
                &[
                    -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                ],
                &[3, 4],
            )],
            &[
                -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
            ],
            &[3, 4],
        );
        check(
            "ij->",
            &[(
                &[
                    -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                ],
                &[3, 4],
            )],
            &[-3.0],
            &[],
        );
        check(
            "ij->j",
            &[(
                &[
                    -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                ],
                &[3, 4],
            )],
            &[-2.0, -5.0, 4.0, 0.0],
            &[4],
        );
        check(
            "ij->i",
            &[(
                &[
                    -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                ],
                &[3, 4],
            )],
            &[-3.0, 0.0, 0.0],
            &[3],
        );
        check(
            "i,i->",
            &[
                (&[1.0, 2.0, -2.0, 1.0], &[4]),
                (&[3.0, 3.0, 3.0, 4.0], &[4]),
            ],
            &[7.0],
            &[],
        );
        check(
            "i,i",
            &[
                (&[1.0, 2.0, -2.0, 1.0], &[4]),
                (&[3.0, 3.0, 3.0, 4.0], &[4]),
            ],
            &[7.0],
            &[],
        );
        check(
            "i,j->ij",
            &[(&[-3.0, 3.0, 0.0], &[3]), (&[1.0, 2.0, -2.0, 1.0], &[4])],
            &[
                -3.0, -6.0, 6.0, -3.0, 3.0, 6.0, -6.0, 3.0, 0.0, 0.0, 0.0, 0.0,
            ],
            &[3, 4],
        );
        check(
            "i,j",
            &[(&[-3.0, 3.0, 0.0], &[3]), (&[1.0, 2.0, -2.0, 1.0], &[4])],
            &[
                -3.0, -6.0, 6.0, -3.0, 3.0, 6.0, -6.0, 3.0, 0.0, 0.0, 0.0, 0.0,
            ],
            &[3, 4],
        );
        check(
            "i,i,i->i",
            &[
                (&[1.0, 2.0, -2.0, 1.0], &[4]),
                (&[3.0, 3.0, 3.0, 4.0], &[4]),
                (&[1.0, 2.0, -2.0, 1.0], &[4]),
            ],
            &[3.0, 12.0, 12.0, 4.0],
            &[4],
        );
        check(
            "i,i,i",
            &[
                (&[1.0, 2.0, -2.0, 1.0], &[4]),
                (&[3.0, 3.0, 3.0, 4.0], &[4]),
                (&[1.0, 2.0, -2.0, 1.0], &[4]),
            ],
            &[31.0],
            &[],
        );
        check(
            "ijk->kji",
            &[(
                &[
                    2.0, -2.0, -4.0, -4.0, 4.0, 4.0, -2.0, -1.0, -2.0, -3.0, 3.0, 2.0, -4.0, -3.0,
                    1.0, 4.0, 4.0, -3.0, 1.0, -4.0, -3.0, -3.0, -1.0, -1.0,
                ],
                &[2, 3, 4],
            )],
            &[
                2.0, -4.0, 4.0, 4.0, -2.0, -3.0, -2.0, -3.0, 4.0, -3.0, -3.0, -3.0, -4.0, 1.0,
                -2.0, 1.0, 3.0, -1.0, -4.0, 4.0, -1.0, -4.0, 2.0, -1.0,
            ],
            &[4, 3, 2],
        );
        check(
            "bij,bjk->bik",
            &[
                (
                    &[
                        2.0, 0.0, -1.0, 4.0, 1.0, 2.0, 2.0, -1.0, 4.0, -4.0, -2.0, -3.0, -4.0, 4.0,
                        -4.0, 0.0, 3.0, 2.0, 3.0, -4.0, 0.0, -4.0, -3.0, 3.0,
                    ],
                    &[2, 3, 4],
                ),
                (
                    &[
                        -4.0, 1.0, -4.0, -2.0, 4.0, -2.0, -2.0, -4.0, 2.0, -3.0, 0.0, -4.0, -4.0,
                        3.0, -4.0, 0.0, -1.0, -3.0, 1.0, 2.0, 4.0, -3.0, 4.0, -4.0, -3.0, 1.0, 3.0,
                        4.0, 2.0, -4.0, -1.0, 3.0, 0.0, -3.0, 2.0, -4.0, 2.0, -4.0, -1.0, -2.0,
                    ],
                    &[2, 4, 5],
                ),
            ],
            &[
                -8.0, 2.0, -16.0, -3.0, 20.0, -8.0, -10.0, -17.0, 7.0, -12.0, -8.0, 23.0, 17.0,
                -25.0, 30.0, -8.0, 12.0, 0.0, 36.0, -12.0, 27.0, -2.0, 36.0, -13.0, -3.0, -13.0,
                -15.0, -28.0, -2.0, 4.0,
            ],
            &[2, 3, 5],
        );
        check(
            "bii->b",
            &[(
                &[
                    4.0, 2.0, 1.0, 0.0, -2.0, 3.0, 4.0, -3.0, -2.0, -2.0, 0.0, -2.0, -1.0, 4.0,
                    -1.0, 4.0, 0.0, -1.0,
                ],
                &[2, 3, 3],
            )],
            &[0.0, 1.0],
            &[2],
        );
        check(
            "bii->bi",
            &[(
                &[
                    4.0, -1.0, 2.0, -2.0, 4.0, 2.0, 3.0, -4.0, -4.0, -4.0, 0.0, -1.0, 4.0, -2.0,
                    -3.0, 3.0, -2.0, 2.0,
                ],
                &[2, 3, 3],
            )],
            &[4.0, 4.0, -4.0, -4.0, -2.0, 2.0],
            &[2, 3],
        );
        check(
            "ijj->i",
            &[(
                &[
                    -2.0, 0.0, -4.0, 1.0, 4.0, 2.0, 3.0, 3.0, 1.0, 4.0, 0.0, -3.0, 1.0, 1.0, -3.0,
                    -3.0, 1.0, -1.0,
                ],
                &[2, 3, 3],
            )],
            &[3.0, 4.0],
            &[2],
        );
        check(
            "iij",
            &[(
                &[
                    0.0, 3.0, 2.0, 4.0, -3.0, 2.0, 0.0, -4.0, -4.0, -1.0, 1.0, -3.0, 0.0, 4.0,
                    -3.0, -3.0, 4.0, -2.0,
                ],
                &[3, 3, 2],
            )],
            &[0.0, 0.0],
            &[2],
        );
        check(
            "iii->i",
            &[(
                &[
                    3.0, -1.0, 2.0, -4.0, 1.0, 3.0, -1.0, 1.0, -3.0, -3.0, -4.0, 0.0, 0.0, -4.0,
                    -2.0, 1.0, -1.0, -2.0, -4.0, -4.0, 1.0, -3.0, 3.0, 0.0, -2.0, 1.0, -3.0,
                ],
                &[3, 3, 3],
            )],
            &[3.0, -4.0, -3.0],
            &[3],
        );
        check(
            " ij , jk -> ik ",
            &[
                (
                    &[
                        -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                    ],
                    &[3, 4],
                ),
                (
                    &[
                        0.0, -4.0, 0.0, -3.0, 2.0, 4.0, 4.0, 1.0, 3.0, -1.0, -3.0, 0.0, -1.0, 1.0,
                        4.0, -2.0, 3.0, -3.0, -1.0, 3.0,
                    ],
                    &[4, 5],
                ),
            ],
            &[
                -21.0, 0.0, -6.0, 3.0, 9.0, 6.0, -12.0, 11.0, 6.0, -3.0, -17.0, 0.0, -14.0, -14.0,
                11.0,
            ],
            &[3, 5],
        );
        check(
            "Ij,jK->IK",
            &[
                (
                    &[
                        -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                    ],
                    &[3, 4],
                ),
                (
                    &[
                        0.0, -4.0, 0.0, -3.0, 2.0, 4.0, 4.0, 1.0, 3.0, -1.0, -3.0, 0.0, -1.0, 1.0,
                        4.0, -2.0, 3.0, -3.0, -1.0, 3.0,
                    ],
                    &[4, 5],
                ),
            ],
            &[
                -21.0, 0.0, -6.0, 3.0, 9.0, 6.0, -12.0, 11.0, 6.0, -3.0, -17.0, 0.0, -14.0, -14.0,
                11.0,
            ],
            &[3, 5],
        );
        check(
            "ij,ij->ij",
            &[
                (&[-2.0, 0.0, 1.0], &[3, 1]),
                (
                    &[
                        -3.0, -1.0, -4.0, -3.0, -2.0, -2.0, 4.0, -1.0, -4.0, 4.0, 1.0, -3.0,
                    ],
                    &[3, 4],
                ),
            ],
            &[6.0, 2.0, 8.0, 6.0, 0.0, 0.0, 0.0, 0.0, -4.0, 4.0, 1.0, -3.0],
            &[3, 4],
        );
        check(
            "ij,ij->i",
            &[
                (&[-4.0, -4.0, 4.0], &[3, 1]),
                (
                    &[
                        1.0, 2.0, 4.0, -1.0, 4.0, -1.0, 2.0, 3.0, -4.0, 2.0, 3.0, 4.0,
                    ],
                    &[3, 4],
                ),
            ],
            &[-24.0, -32.0, 20.0],
            &[3],
        );
        check(
            "i,i->i",
            &[(&[2.0], &[1]), (&[1.0, 2.0, -2.0, 1.0], &[4])],
            &[2.0, 4.0, -4.0, 2.0],
            &[4],
        );
        check(
            "...ij,...jk->...ik",
            &[
                (
                    &[
                        0.0, -3.0, 2.0, 0.0, 3.0, -4.0, -1.0, 3.0, 1.0, 2.0, 1.0, 3.0, 3.0, 2.0,
                        1.0, -2.0, -4.0, 3.0, -1.0, -2.0, -4.0, -3.0, -4.0, -1.0,
                    ],
                    &[2, 3, 4],
                ),
                (
                    &[
                        -2.0, 0.0, 1.0, -2.0, 0.0, 1.0, 2.0, 1.0, -4.0, -2.0, 3.0, 1.0, -2.0, -1.0,
                        4.0, 2.0, -1.0, -2.0, -3.0, 3.0, 1.0, -1.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0,
                        -1.0, -2.0, -2.0, 4.0, -3.0, -4.0, 1.0, -2.0, -4.0, 0.0, 2.0, -4.0,
                    ],
                    &[2, 4, 5],
                ),
            ],
            &[
                3.0, -4.0, -7.0, 10.0, 14.0, -7.0, -12.0, -5.0, 2.0, 13.0, 9.0, 2.0, -5.0, -20.0,
                9.0, 9.0, 13.0, 3.0, -10.0, 5.0, 8.0, 14.0, -5.0, -3.0, 1.0, 0.0, -14.0, 4.0, 17.0,
                6.0,
            ],
            &[2, 3, 5],
        );
        check(
            "...ij,...jk->...ik",
            &[
                (
                    &[0.0, -3.0, 1.0, 3.0, 3.0, 3.0, -2.0, 1.0, 1.0, 4.0, 2.0, 2.0],
                    &[3, 4],
                ),
                (
                    &[
                        -4.0, -2.0, -1.0, 1.0, 3.0, -1.0, 0.0, 2.0, -3.0, -1.0, -3.0, -3.0, 2.0,
                        3.0, -3.0, 2.0, 2.0, 2.0, 2.0, 1.0, -3.0, 3.0, -3.0, 0.0, 4.0, 1.0, -2.0,
                        -4.0, -2.0, 0.0, 3.0, 3.0, -3.0, 2.0, 1.0, 3.0, 0.0, 0.0, -1.0, 3.0,
                    ],
                    &[2, 4, 5],
                ),
            ],
            &[
                6.0, 3.0, 2.0, 18.0, 3.0, -7.0, 2.0, 1.0, -10.0, 13.0, -10.0, -4.0, 15.0, -1.0,
                -5.0, 9.0, 9.0, 9.0, 5.0, 10.0, -9.0, -3.0, -15.0, -11.0, 13.0, 13.0, 1.0, -25.0,
                -6.0, 12.0,
            ],
            &[2, 3, 5],
        );
        check(
            "...ij,jk->...ik",
            &[
                (
                    &[
                        0.0, -4.0, 0.0, 3.0, 3.0, -1.0, 3.0, -4.0, 2.0, 1.0, -4.0, 0.0, -1.0, -1.0,
                        -2.0, 3.0, 3.0, -3.0, 2.0, 1.0, -3.0, -1.0, -2.0, 0.0,
                    ],
                    &[2, 3, 4],
                ),
                (
                    &[
                        0.0, -4.0, 0.0, -3.0, 2.0, 4.0, 4.0, 1.0, 3.0, -1.0, -3.0, 0.0, -1.0, 1.0,
                        4.0, -2.0, 3.0, -3.0, -1.0, 3.0,
                    ],
                    &[4, 5],
                ),
            ],
            &[
                -22.0, -7.0, -13.0, -15.0, 13.0, -5.0, -28.0, 8.0, -5.0, 7.0, 16.0, -4.0, 5.0,
                -7.0, -13.0, -4.0, 9.0, -8.0, -5.0, 0.0, -20.0, -21.0, -8.0, -17.0, 20.0, 2.0, 8.0,
                1.0, 4.0, -13.0,
            ],
            &[2, 3, 5],
        );
        check(
            "...ij,jk",
            &[
                (
                    &[
                        -3.0, 0.0, -3.0, -3.0, 3.0, 0.0, -2.0, 3.0, -1.0, -3.0, 4.0, -4.0, 4.0,
                        -4.0, 3.0, 0.0, -3.0, 4.0, -1.0, -4.0, 2.0, 4.0, -3.0, 0.0,
                    ],
                    &[2, 3, 4],
                ),
                (
                    &[
                        0.0, -4.0, 0.0, -3.0, 2.0, 4.0, 4.0, 1.0, 3.0, -1.0, -3.0, 0.0, -1.0, 1.0,
                        4.0, -2.0, 3.0, -3.0, -1.0, 3.0,
                    ],
                    &[4, 5],
                ),
            ],
            &[
                15.0, 3.0, 12.0, 9.0, -27.0, 0.0, -3.0, -7.0, -14.0, 7.0, -16.0, -20.0, 5.0, 2.0,
                5.0, -25.0, -32.0, -7.0, -21.0, 24.0, 27.0, 16.0, 17.0, 24.0, -26.0, 25.0, 8.0,
                7.0, 3.0, -12.0,
            ],
            &[2, 3, 5],
        );
        check(
            "...->...",
            &[(&[4.0, -3.0, 3.0, -1.0, -4.0, -3.0], &[2, 3])],
            &[4.0, -3.0, 3.0, -1.0, -4.0, -3.0],
            &[2, 3],
        );
        check(
            "...",
            &[(&[3.0, 2.0, 3.0, 0.0, -3.0, -2.0], &[2, 3])],
            &[3.0, 2.0, 3.0, 0.0, -3.0, -2.0],
            &[2, 3],
        );
        check(
            "i...->...i",
            &[(
                &[
                    -2.0, -2.0, -4.0, 3.0, -4.0, 2.0, 2.0, -1.0, 1.0, -1.0, 0.0, 2.0, 0.0, 4.0,
                    3.0, -4.0, 4.0, -3.0, -3.0, -1.0, 3.0, 0.0, 2.0, -4.0,
                ],
                &[2, 3, 4],
            )],
            &[
                -2.0, 0.0, -2.0, 4.0, -4.0, 3.0, 3.0, -4.0, -4.0, 4.0, 2.0, -3.0, 2.0, -3.0, -1.0,
                -1.0, 1.0, 3.0, -1.0, 0.0, 0.0, 2.0, 2.0, -4.0,
            ],
            &[3, 4, 2],
        );
        check(
            "i...,i...->...",
            &[
                (&[2.0, -4.0, -2.0, -3.0, 3.0, -1.0], &[2, 3]),
                (&[2.0, 2.0, 2.0, 1.0, 3.0, -4.0], &[2, 3]),
            ],
            &[1.0, 1.0, 0.0],
            &[3],
        );
        check(
            "...,...->...",
            &[(&[1.0, -4.0], &[2, 1]), (&[1.0, 0.0, 4.0], &[1, 3])],
            &[1.0, 0.0, 4.0, -4.0, 0.0, -16.0],
            &[2, 3],
        );
        check(",->", &[(&[2.0], &[]), (&[3.0], &[])], &[6.0], &[]);
        check(
            "i,->i",
            &[(&[1.0, 2.0, -2.0, 1.0], &[4]), (&[2.0], &[])],
            &[2.0, 4.0, -4.0, 2.0],
            &[4],
        );
        check("->", &[(&[5.0], &[])], &[5.0], &[]);
        check(
            "ij,j",
            &[
                (
                    &[
                        -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                    ],
                    &[3, 4],
                ),
                (&[1.0, 2.0, -2.0, 1.0], &[4]),
            ],
            &[-15.0, -5.0, 0.0],
            &[3],
        );
        check(
            "ij,i",
            &[
                (
                    &[
                        -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                    ],
                    &[3, 4],
                ),
                (&[-3.0, 3.0, 0.0], &[3]),
            ],
            &[12.0, 12.0, -3.0, -12.0],
            &[4],
        );
        check(
            "ab,ab->",
            &[
                (
                    &[
                        -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                    ],
                    &[3, 4],
                ),
                (
                    &[
                        -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                    ],
                    &[3, 4],
                ),
            ],
            &[75.0],
            &[],
        );
        check(
            "ab,cb->ac",
            &[
                (
                    &[
                        -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
                    ],
                    &[3, 4],
                ),
                (&[3.0, -4.0, 4.0, -3.0, -1.0, 0.0, 3.0, -1.0], &[2, 4]),
            ],
            &[15.0, 12.0, 19.0, 9.0, -4.0, -7.0],
            &[3, 2],
        );
        check(
            "ij,kl->ijkl",
            &[
                (&[4.0, -2.0, -1.0, 4.0], &[2, 2]),
                (&[2.0, 2.0, -1.0, -4.0, -3.0, -2.0], &[2, 3]),
            ],
            &[
                8.0, 8.0, -4.0, -16.0, -12.0, -8.0, -4.0, -4.0, 2.0, 8.0, 6.0, 4.0, -2.0, -2.0,
                1.0, 4.0, 3.0, 2.0, 8.0, 8.0, -4.0, -16.0, -12.0, -8.0,
            ],
            &[2, 2, 2, 3],
        );
    }

    #[test]
    fn einsum_empty_dimensions() {
        check(
            "ij,jk->ik",
            &[(&[], &[2, 0]), (&[], &[0, 3])],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[2, 3],
        );
        check("ij->", &[(&[], &[2, 0])], &[0.0], &[]);
        check("ij->i", &[(&[], &[0, 3])], &[], &[0]);
        check("i,i->", &[(&[], &[0]), (&[], &[0])], &[0.0], &[]);
    }

    #[test]
    fn einsum_strided_operands() {
        // A is 3x4 stored row-major; take its transpose as a strided view (strides [1, 4])
        let a = [
            -3.0, -3.0, 3.0, 0.0, 1.0, 1.0, 2.0, -4.0, 0.0, -3.0, -1.0, 4.0,
        ];
        let at = Operand {
            data: &a,
            shape: &[4, 3],
            strides: &[1, 4],
            offset: 0,
        };
        let b = [
            0.0, -4.0, 0.0, -3.0, 2.0, 4.0, 4.0, 1.0, 3.0, -1.0, -3.0, 0.0, -1.0, 1.0, 4.0, -2.0,
            3.0, -3.0, -1.0, 3.0,
        ];
        let bo = Operand {
            data: &b,
            shape: &[4, 5],
            strides: &c_strides(&[4, 5]),
            offset: 0,
        };
        let (got, shape) = einsum("ji,jk->ik", &[at, bo]);
        assert_eq!(shape, vec![3, 5]);
        assert_close(
            &got,
            &[
                -21.0, 0.0, -6.0, 3.0, 9.0, 6.0, -12.0, 11.0, 6.0, -3.0, -17.0, 0.0, -14.0, -14.0,
                11.0,
            ],
        );
        // offset + negative stride: reversed view of a vector
        let x = [1.0, 2.0, 3.0, 4.0];
        let rev = Operand {
            data: &x,
            shape: &[4],
            strides: &[-1],
            offset: 3,
        };
        let fwd = Operand {
            data: &x,
            shape: &[4],
            strides: &[1],
            offset: 0,
        };
        let (got, _) = einsum("i,i->", &[rev, fwd]);
        assert_eq!(got, vec![20.0]);
        // broadcast (stride 0) operand
        let y = [2.0, 3.0];
        let bc = Operand {
            data: &y,
            shape: &[3, 2],
            strides: &[0, 1],
            offset: 0,
        };
        let (got, shape) = einsum("ij->j", &[bc]);
        assert_eq!(shape, vec![2]);
        assert_eq!(got, vec![6.0, 9.0]);
    }

    #[test]
    fn einsum_integers() {
        let a = [1i64, 2, 3, 4];
        let b = [5i64, 6, 7, 8];
        let oa = Operand {
            data: &a,
            shape: &[2, 2],
            strides: &c_strides(&[2, 2]),
            offset: 0,
        };
        let ob = Operand {
            data: &b,
            shape: &[2, 2],
            strides: &c_strides(&[2, 2]),
            offset: 0,
        };
        let (got, shape) = einsum("ij,jk->ik", &[oa, ob]);
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(got, vec![19, 22, 43, 50]);
    }

    #[test]
    fn einsum_errors() {
        let e = |sub: &str, shapes: &[&[usize]]| {
            let sub = sub.to_string();
            let shapes: Vec<Vec<usize>> = shapes.iter().map(|s| s.to_vec()).collect();
            std::panic::catch_unwind(move || {
                let datas: Vec<Vec<f64>> = shapes
                    .iter()
                    .map(|s| vec![1.0; s.iter().product()])
                    .collect();
                let strides: Vec<Vec<isize>> = shapes.iter().map(|s| c_strides(s)).collect();
                let ops: Vec<Operand<f64>> = (0..shapes.len())
                    .map(|k| Operand {
                        data: &datas[k],
                        shape: &shapes[k],
                        strides: &strides[k],
                        offset: 0,
                    })
                    .collect();
                einsum(&sub, &ops)
            })
            .is_err()
        };
        assert!(e("ij,jk->ik", &[&[2, 3], &[4, 5]]), "size mismatch");
        assert!(e("ij->ij", &[&[2, 3, 4]]), "operand has more dimensions");
        assert!(e("ijk->ij", &[&[2, 3]]), "too many subscripts");
        assert!(e("ij->ik", &[&[2, 3]]), "unknown output label");
        assert!(e("ij->ii", &[&[2, 2]]), "repeated output label");
        assert!(e("i1->i", &[&[2, 2]]), "invalid character");
        assert!(e("ij,jk->ik", &[&[2, 3]]), "too few operands");
        assert!(e("ij->ij", &[&[2, 3], &[2, 3]]), "too many operands");
        assert!(e("ii->i", &[&[2, 3]]), "diagonal of non-square");
        assert!(e("i.j->ij", &[&[2, 3]]), "stray dot");
        assert!(
            e("i...->i", &[&[2, 3, 4]]),
            "ellipsis dims need an output ellipsis"
        );
        assert!(
            !e("ij->i...", &[&[2, 3]]),
            "empty ellipsis in output is allowed"
        );
        assert!(e("ij->i", &[]), "no operands");
    }

    #[test]
    fn einsum_implicit_output_order_and_more_operands() {
        check(
            "bA",
            &[(&[1.0, 2.0, -3.0, 2.0, 0.0, 0.0], &[2, 3])],
            &[1.0, 2.0, 2.0, 0.0, -3.0, 0.0],
            &[3, 2],
        );
        check(
            "aB,Bc",
            &[
                (&[3.0, -2.0, -3.0, -1.0, 1.0, -3.0], &[2, 3]),
                (
                    &[
                        3.0, -1.0, -2.0, 0.0, 3.0, 3.0, 3.0, -1.0, -3.0, 2.0, 1.0, 2.0,
                    ],
                    &[3, 4],
                ),
            ],
            &[12.0, -15.0, -15.0, -4.0, 9.0, -2.0, 2.0, -7.0],
            &[2, 4],
        );
        check(
            "ij,ij,ij->",
            &[
                (&[1.0, -1.0, 3.0, -3.0, -2.0, -1.0], &[2, 3]),
                (&[0.0, -1.0, -3.0, -3.0, -3.0, -3.0], &[2, 3]),
                (&[-2.0, 3.0, -2.0, 1.0, 2.0, -2.0], &[2, 3]),
            ],
            &[36.0],
            &[],
        );
        check(
            "ij,ij,ij",
            &[
                (&[1.0, -1.0, 3.0, -3.0, -2.0, -1.0], &[2, 3]),
                (&[0.0, -1.0, -3.0, -3.0, -3.0, -3.0], &[2, 3]),
                (&[-2.0, 3.0, -2.0, 1.0, 2.0, -2.0], &[2, 3]),
            ],
            &[36.0],
            &[],
        );
        check(
            "ij,jk,ki->",
            &[
                (&[-2.0, 0.0, -2.0, 3.0, -2.0, 3.0, 2.0, 2.0, -3.0], &[3, 3]),
                (&[-1.0, 1.0, 0.0, 1.0, 1.0, -3.0, 0.0, 3.0, -2.0], &[3, 3]),
                (&[-2.0, 0.0, -2.0, 3.0, -2.0, 3.0, 2.0, 2.0, -3.0], &[3, 3]),
            ],
            &[-55.0],
            &[],
        );
        check(
            "ii,ii->",
            &[
                (&[-2.0, 0.0, -2.0, 3.0, -2.0, 3.0, 2.0, 2.0, -3.0], &[3, 3]),
                (&[-2.0, 0.0, -2.0, 3.0, -2.0, 3.0, 2.0, 2.0, -3.0], &[3, 3]),
            ],
            &[17.0],
            &[],
        );
        check(
            "ij,kj->ik",
            &[
                (&[1.0, -1.0, 3.0, -3.0, -2.0, -1.0], &[2, 3]),
                (&[0.0, -1.0, -3.0, -3.0, -3.0, -3.0], &[2, 3]),
            ],
            &[-8.0, -9.0, 5.0, 18.0],
            &[2, 2],
        );
        check(
            "ba,bc",
            &[
                (&[1.0, -1.0, 3.0, -3.0, -2.0, -1.0], &[2, 3]),
                (&[0.0, -1.0, -3.0, -3.0, -3.0, -3.0], &[2, 3]),
            ],
            &[9.0, 8.0, 6.0, 6.0, 7.0, 9.0, 3.0, 0.0, -6.0],
            &[3, 3],
        );
    }
}
