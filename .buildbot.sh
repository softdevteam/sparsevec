#! /bin/sh

set -e

export CARGO_HOME="`pwd`/.cargo"
export RUSTUP_HOME="`pwd`/.rustup"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
sh rustup.sh --default-host x86_64-unknown-linux-gnu --default-toolchain stable -y --no-modify-path

export PATH=`pwd`/.cargo/bin/:$PATH

cargo fmt --all -- --check
RUSTDOCFLAGS="-Dwarnings" cargo doc --no-deps

cargo test
cargo test --release

which cargo-deny || cargo install cargo-deny || true
if [ "X`which cargo-deny`" != "X" ]; then
    cargo-deny check license
fi
