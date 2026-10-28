//! ROCm kernels for candle, compiled from the shared CUDA sources with `hipcc`, and
//! the minimal HIP runtime and rocBLAS bindings the ROCm backend needs.
pub mod ffi;

mod code_objects {
    include!(concat!(env!("OUT_DIR"), "/code_objects.rs"));
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Indexing,
    Quantized,
    Random,
    Reduce,
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 12] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::Indexing,
    Id::Quantized,
    Id::Random,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

/// A compiled kernel module: a clang offload bundle with one code object per
/// target architecture, loadable with `hipModuleLoadData`.
pub struct Module {
    index: usize,
    code: &'static [u8],
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn code(&self) -> &'static [u8] {
        self.code
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            code: code_objects::$cst,
        };
    };
}

mdl!(AFFINE, Affine);
mdl!(BINARY, Binary);
mdl!(CAST, Cast);
mdl!(CONV, Conv);
mdl!(FILL, Fill);
mdl!(INDEXING, Indexing);
mdl!(QUANTIZED, Quantized);
mdl!(RANDOM, Random);
mdl!(REDUCE, Reduce);
mdl!(SORT, Sort);
mdl!(TERNARY, Ternary);
mdl!(UNARY, Unary);
