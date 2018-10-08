# Sparse Vector (SparseVec)

A SparseVec compresses a matrix (passed as a one dimensional vector) using row
displacement as described in [1]. It then uses a PackedVec to further
reduce the size of the result.

## Usage

```
extern crate sparse_vec;
use sparse_vec::SparseVec;

fn main() {
    let v:Vec<usize> = vec![1,2,3,4,5,6,7,8];
    let sv = SparseVec::from(&v, 0, 4);
    assert_eq!(sv.get(1,2).unwrap(), 7);
}

```

[1] Tarjan, Robert Endre, and Andrew Chi-Chih Yao. "Storing a sparse table."
Communications of the ACM 22.11 (1979): 606-611.
