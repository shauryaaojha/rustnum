#[cfg(feature = "rayon")]
pub(crate) const PAR_THRESHOLD: usize = 1 << 15;
pub(crate) fn size(shape: &[usize]) -> usize {
    if shape.contains(&0) {
        return 0;
    }
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .expect("array is too big")
}
pub(crate) fn strides(shape: &[usize]) -> Vec<isize> {
    let mut out = vec![0; shape.len()];
    let mut n = 1isize;
    for (i, &d) in shape.iter().enumerate().rev() {
        out[i] = n;
        n = n
            .checked_mul(isize::try_from(d).expect("array is too big"))
            .expect("array is too big");
    }
    out
}
pub(crate) fn contiguous(shape: &[usize], st: &[isize]) -> bool {
    if shape.contains(&0) {
        return true;
    }
    let mut expected = 1;
    for (&d, &s) in shape.iter().zip(st).rev() {
        if d > 1 && s != expected {
            return false;
        }
        expected *= d as isize;
    }
    true
}
pub(crate) struct Offsets {
    shape: Vec<usize>,
    strides: Vec<isize>,
    coord: Vec<usize>,
    offset: isize,
    left: usize,
}
impl Offsets {
    pub(crate) fn new(shape: &[usize], strides: &[isize], offset: usize) -> Self {
        Self {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            coord: vec![0; shape.len()],
            offset: offset as isize,
            left: size(shape),
        }
    }
}
impl Iterator for Offsets {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.left == 0 {
            return None;
        }
        let out = self.offset as usize;
        self.left -= 1;
        if self.left > 0 {
            for i in (0..self.shape.len()).rev() {
                self.coord[i] += 1;
                if self.coord[i] < self.shape[i] {
                    self.offset += self.strides[i];
                    break;
                }
                self.coord[i] = 0;
                self.offset -= self.strides[i] * (self.shape[i] as isize - 1);
            }
        }
        Some(out)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}
impl ExactSizeIterator for Offsets {}
pub(crate) fn advance(coord: &mut [usize], shape: &[usize]) {
    for i in (0..shape.len()).rev() {
        coord[i] += 1;
        if coord[i] < shape[i] {
            break;
        }
        coord[i] = 0;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reverse_offsets() {
        assert_eq!(
            Offsets::new(&[2, 3], &[3, -1], 2).collect::<Vec<_>>(),
            [2, 1, 0, 5, 4, 3]
        );
    }
    #[test]
    fn singleton_contiguity() {
        assert!(contiguous(&[2, 1, 3], &[3, 0, 1]));
    }
    #[test]
    fn empty_offsets() {
        assert_eq!(Offsets::new(&[0], &[-1], 0).next(), None);
    }
}
