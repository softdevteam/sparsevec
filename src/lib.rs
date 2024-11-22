#![allow(clippy::many_single_char_names)]

use num_traits::{AsPrimitive, FromPrimitive, PrimInt, ToPrimitive, Unsigned};
use packedvec::PackedVec;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use vob::Vob;

/// A SparseVec efficiently encodes a two-dimensional matrix of integers. The input matrix must be
/// encoded as a one-dimensional vector of integers with a row-length. Given an "empty" value, the
/// SparseVec uses row displacement to compress that value as described in "Storing a sparse table"
/// by Robert Endre Tarjan and Andrew Chi-Chih Yao. Afterwards it encodes the result further using
/// a PackedVec.
///
/// # Example
///
/// ```
/// extern crate sparsevec;
/// use sparsevec::SparseVec;
///
/// fn main() {
///     let v:Vec<usize> = vec![1,0,0,0,
///                             0,0,7,8,
///                             9,0,0,3];
///     let sv = SparseVec::from(&v, 0, 4);
///     assert_eq!(sv.get(0,0).unwrap(), 1);
///     assert_eq!(sv.get(1,2).unwrap(), 7);
///     assert_eq!(sv.get(2,3).unwrap(), 3);
/// }
/// ```
///
/// # How it works
///
/// Let's take as an example the two-dimensional vector
/// ```text
/// 1 0 0
/// 2 0 0
/// 3 0 0
/// 0 0 4
/// ```
/// represented as a one dimensional vector `v = [1,0,0,2,0,0,3,0,0,0,0,4]` with row-length 3.
/// Storing this vector in memory is wasteful as the majority of its elements is 0. We can compress
/// this vector using row displacement, which merges all rows into a vector such that non-zero
/// entries are never mapped to the same position. For the above example, this would result in the
/// compressed vector `c = [1,2,3,0,4]`:
/// ```text
/// 1 0 0
///   2 0 0
///     3 0 0
///     0 0 4
/// ---------
/// 1 2 3 0 4
/// ```
/// To retrieve values from the compressed vector, we need a displacement vector, which
/// describes how much each row was shifted during the compression. For the above example, the
/// displacement vector would be `d = [0, 1, 2, 2]`. In order to retrieve the value at
/// position (2, 0), we can calculate its compressed position with `pos = d[row] + col`:
/// ```text
/// pos = d[2] + 0 // =2
/// value = c[pos] // =3
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct SparseVec<T> {
    displacement: Vec<usize>, // Displacement vector
    row_length: usize,        // Row length of the input matrix
    empty_val: T,             // Value considered "empty"
    empties: Vob<u64>,        // Mapping of "empty" cells
    data: PackedVec<T, u64>,  // Compressed matrix
}

impl<T: Clone + Copy + PartialEq> SparseVec<T>
where
    T: AsPrimitive<u64> + FromPrimitive + Ord + PrimInt + ToPrimitive + Unsigned,
    u64: AsPrimitive<T>,
{
    /// Constructs a new SparseVec from a `Vec` of unsigned integers where `empty_val` describes
    /// the values to be compressed and `row_length` the element size per row in the original
    /// two-dimensional vector.
    ///
    /// # Examples
    /// ```
    /// use sparsevec::SparseVec;
    /// let v:Vec<usize> = vec![1,2,3,4,5,6,7,8];
    /// let sv = SparseVec::from(&v, 0, 4);
    /// assert_eq!(sv.get(1,2).unwrap(), 7);
    /// ```
    pub fn from(v: &[T], empty_val: T, row_length: usize) -> SparseVec<T> {
        if v.is_empty() {
            return SparseVec {
                displacement: Vec::new(),
                row_length: 0,
                empty_val,
                empties: Vob::<u64>::new_with_storage_type(0),
                data: PackedVec::<T, u64>::new_with_storaget(v.to_vec()),
            };
        }

        // Sort rows by amount of empty values as suggested in
        // "Smaller  faster  table  driven  parser" by S. F. Zeigler
        let s = sort(v, empty_val, row_length);
        let (c, d) = compress(v, &s, empty_val, row_length);
        let e = calc_empties(v, empty_val);
        let pv = PackedVec::<T, u64>::new_with_storaget(c);
        SparseVec {
            displacement: d,
            row_length,
            empty_val,
            empties: e,
            data: pv,
        }
    }

    /// Returns the value of the element at position `(r,c)`, where `r` is a row and `c` is a
    /// column. Returns `None` if out of bounds.
    ///
    /// # Examples
    /// ```
    /// use sparsevec::SparseVec;
    /// let v:Vec<usize> = vec![1,2,3,4,5,6,7,8];
    /// let sv = SparseVec::from(&v, 0, 4);
    /// assert_eq!(sv.get(1,2).unwrap(), 7);
    /// ```
    pub fn get(&self, r: usize, c: usize) -> Option<T> {
        let k = r * self.row_length + c;
        match self.empties.get(k) {
            None => None,
            Some(true) => Some(self.empty_val),
            Some(false) => self.data.get(self.displacement[r] + c),
        }
    }

    /// Returns the number of elements of the original input vector.
    /// # Examples
    /// ```
    /// use sparsevec::SparseVec;
    /// let v = vec![1,2,3,4];
    /// let sv = SparseVec::from(&v, 0 as usize, 2);
    /// assert_eq!(sv.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.empties.len()
    }

    /// Returns true if the SparseVec has no elements or false otherwise.
    /// # Examples
    /// ```
    /// use sparsevec::SparseVec;
    /// let v = Vec::new();
    /// let sv = SparseVec::from(&v, 0 as usize, 0);
    /// assert_eq!(sv.is_empty(), true);
    /// ```
    pub fn is_empty(&self) -> bool {
        self.empties.is_empty()
    }
}

fn calc_empties<T: PartialEq>(vec: &[T], empty_val: T) -> Vob<u64> {
    let mut vob = Vob::<u64>::from_elem_with_storage_type(false, vec.len());
    for (i, v) in vec.iter().enumerate() {
        if *v == empty_val {
            vob.set(i, true);
        }
    }
    vob
}

fn compress<T: Clone + Copy + PartialEq>(
    vec: &[T],
    sorted: &[usize],
    empty_val: T,
    row_length: usize,
) -> (Vec<T>, Vec<usize>) {
    let mut r = Vec::new(); // Result vector
    r.resize(row_length, empty_val);

    let mut dv = vec![0; sorted.len()]; // displacement vector

    let mut tmp = Vec::new();
    for s in sorted {
        // The row we're about to iterate over typically contains mostly empty values that can
        // never succeed with `fits`. We pre-filter out all those empty values up-front, such that
        // `tmp` contains `(index, non-empty-value)` pairs that we can then pass to `fits`. Because
        // this is such a tight loop, we reuse the same `Vec` to avoid repeated allocations.
        tmp.clear();
        tmp.extend(
            vec[s * row_length..(s + 1) * row_length]
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != empty_val),
        );

        let mut d = 0; // displacement value
        loop {
            if fits(tmp.as_slice(), &r, d, empty_val) {
                apply(tmp.as_slice(), &mut r, d);
                dv[*s] = d;
                break;
            } else {
                d += 1;
                if d + row_length > r.len() {
                    r.resize(d + row_length, empty_val); // increase result vector size
                }
            }
        }
    }
    (r, dv)
}

/// `v` is an array of `(index, non-empty_val)` pairs.
fn fits<T: PartialEq>(v: &[(usize, &T)], target: &[T], d: usize, empty_val: T) -> bool {
    for (i, x) in v {
        if target[d + i] != empty_val && target[d + i] != **x {
            return false;
        }
    }
    true
}

/// `v` is an array of `(index, non-empty_val)` pairs.
fn apply<T: Copy + PartialEq>(v: &[(usize, &T)], target: &mut [T], d: usize) {
    for (i, x) in v {
        target[d + i] = **x;
    }
}

fn sort<T: PartialEq>(v: &[T], empty_val: T, row_length: usize) -> Vec<usize> {
    let mut o: Vec<usize> = (0..v.len() / row_length).collect();
    o.sort_by_key(|x| {
        v[(x * row_length)..((x + 1) * row_length)]
            .iter()
            .filter(|y| *y == &empty_val)
            .count()
    });
    o
}

#[cfg(test)]
mod test {
    extern crate rand;
    use super::*;

    #[test]
    fn test_sparsevec() {
        let v = vec![0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 5, 6, 0, 7, 8, 0];
        let sv = SparseVec::from(&v, 0 as usize, 4);
        assert_eq!(sv.get(0, 0).unwrap(), 0);
        assert_eq!(sv.get(0, 1).unwrap(), 1);
        assert_eq!(sv.get(0, 2).unwrap(), 2);
        assert_eq!(sv.get(0, 3).unwrap(), 3);
        assert_eq!(sv.get(1, 0).unwrap(), 4);
        assert_eq!(sv.get(1, 1).unwrap(), 0);
        assert_eq!(sv.get(2, 2).unwrap(), 5);
        assert_eq!(sv.get(2, 3).unwrap(), 6);
        assert_eq!(sv.get(3, 0).unwrap(), 0);
        assert_eq!(sv.get(3, 1).unwrap(), 7);
        assert_eq!(sv.get(3, 2).unwrap(), 8);
        assert_eq!(sv.get(3, 3).unwrap(), 0);
    }

    #[test]
    fn test_sparsevec_empty() {
        let v = Vec::new();
        let sv = SparseVec::from(&v, 0 as usize, 0);
        assert_eq!(sv.len(), 0);
        assert_eq!(sv.get(0, 0), None);
        assert_eq!(sv.is_empty(), true);
    }

    fn random_sparsevec(row_length: usize) {
        const LENGTH: usize = 2000;

        let mut v: Vec<u16> = Vec::with_capacity(LENGTH);
        for _ in 0..LENGTH {
            if rand::random::<u8>() < 128 {
                v.push(0);
            } else {
                v.push(rand::random::<u16>() % 1000);
            }
        }

        let sv = SparseVec::from(&v, 0, row_length);
        let rows = LENGTH / row_length;
        for r in 0..rows {
            for c in 0..row_length {
                assert_eq!(sv.get(r, c).unwrap(), v[r * row_length + c]);
            }
        }
    }

    #[test]
    fn random_vec() {
        random_sparsevec(5);
        random_sparsevec(10);
        random_sparsevec(20);
        random_sparsevec(50);
        random_sparsevec(100);
    }

    #[test]
    fn test_sparsevec_compress_same_values() {
        let v = vec![0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 0, 0, 1, 2, 0];

        let s: Vec<usize> = (0..v.len() / 4).collect();
        let (c, d) = compress(&v, &s, 0 as usize, 4);
        assert_eq!(c, vec![0, 1, 2, 3, 0]);
        assert_eq!(d, vec![0, 0, 1, 0]);

        let sv = SparseVec::from(&v, 0 as usize, 4);
        assert_eq!(sv.get(0, 0).unwrap(), 0);
        assert_eq!(sv.get(0, 1).unwrap(), 1);
        assert_eq!(sv.get(0, 2).unwrap(), 2);
        assert_eq!(sv.get(0, 3).unwrap(), 3);
        assert_eq!(sv.get(1, 0).unwrap(), 0);
        assert_eq!(sv.get(1, 1).unwrap(), 1);
        assert_eq!(sv.get(2, 0).unwrap(), 1);
        assert_eq!(sv.get(2, 1).unwrap(), 2);
        assert_eq!(sv.get(2, 2).unwrap(), 3);
        assert_eq!(sv.get(2, 3).unwrap(), 0);
        assert_eq!(sv.get(3, 0).unwrap(), 0);
        assert_eq!(sv.get(3, 1).unwrap(), 1);
        assert_eq!(sv.get(3, 2).unwrap(), 2);
        assert_eq!(sv.get(3, 3).unwrap(), 0);
    }

    #[test]
    fn test_sort_function() {
        let v = vec![1, 0, 0, 0, 8, 9, 0, 0, 5, 6, 7, 0, 1, 2, 3, 4];
        let s = sort(&v, 0, 4);
        assert_eq!(s, [3, 2, 1, 0]);

        let v = vec![1, 0, 1, 0, 0, 1, 0, 0, 8, 9, 0, 0, 0, 2, 3, 4];
        let s = sort(&v, 0, 4);
        assert_eq!(s, [3, 0, 2, 1]);
    }
}
