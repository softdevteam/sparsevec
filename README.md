# Sparse Vector (SparseVec)

A SparseVec efficiently encodes a two-dimensional matrix of integers. The input
matrix must be encoded as a one-dimensional vector of integers with a
row-length. Given an empty value, the SparseVec uses row displacement as
described in [1] for the compression and encodes the result further using a
PackedVec.

[1] Tarjan, Robert Endre, and Andrew Chi-Chih Yao. "Storing a sparse table."
Communications of the ACM 22.11 (1979): 606-611.

# Usage

```rust
extern crate sparsevec;
use sparsevec::SparseVec;

fn main() {
    use sparsevec::SparseVec;
    let v:Vec<usize> = vec![1,0,0,0,
                            0,0,7,8,
                            9,0,0,3];
    let sv = SparseVec::from(&v, 0, 4);
    assert_eq!(sv.get(0,0).unwrap(), 1);
    assert_eq!(sv.get(1,2).unwrap(), 7);
    assert_eq!(sv.get(2,3).unwrap(), 3);
}
```

# How it works

Let's take as an example the two-dimensional vector
```
1 0 0
2 0 0
3 0 0
0 0 4
```
represented as a one dimensional vector `v = [1,0,0,2,0,0,3,0,0,0,0,4]` with row-length 3.
Storing this vector in memory is wasteful as the majority of its elements is 0. We can compress
this vector using row displacement, which merges all rows into a vector such that no two
non-zero entries are mapped to the same position. For the above example, this would result in
the compressed vector `c = [1,2,3,0,4]`:
```
1 0 0
  2 0 0
    3 0 0
    0 0 4
---------
1 2 3 0 4
```
To retrieve values from the compressed vector, we need a displacement vector, which
describes how much each row was shifted during the compression. For the above example, the
displacement vector would be `d = [0, 1, 2, 2]`. In order to retrieve the value at
position (2, 0), we can calculate its compressed position with `pos = d[row] + col`:
```
pos = d[2] + 0 // =2
value = c[pos] // =3
```
