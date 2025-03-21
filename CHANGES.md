# sparsevec 0.2.2 (2025-03-18)

* Add an optional bincode dependency, which can be used as an alternative to
  serde for encoding / decoding.


# sparsevec 0.2.1 (2024-11-24)

* Optimise the creation of a sparsevec: when used, for example, for grmtools'
  grammars, this can be a substantial (e.g. in one example over 30x) speedup.


# sparsevec 0.2.0 (2022-07-25)

* Move the backing storage (consisting of `PackedVec`s and `Vob`s) from `usize`
  to `u64`. This makes serialising/deserialising across machine widths possible
  (though not necessarily reliable!).


# sparsevec 0.1.4 (2021-10-20)

* Upgrade dependencies.


# sparsevec 0.1.3 (2019-12-06)

* Allow `SparseVec`s to be printed with the debug formatter `{:?}`.


# sparsevec 0.1.2 (2019-11-21)

* License as dual Apache-2.0/MIT (instead of a more complex, and little
  understood, triple license of Apache-2.0/MIT/UPL-1.0).


# sparsevec 0.1.1 (2019-05-08)

* Make `rand` a `dev-dependency` to cut compile times down.

* Migrate to Rust 2018.

* Minor code improvements, including functions now accepting `&[T]` slices
  rather than `&Vec<T>` vectors, making the API a little more flexible to use.


# sparsevec 0.1.0 (2018-10-17)

Initial release.
