use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

use memmap2::Mmap;
use ndarray::{Array1, Array2, ArrayView2};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};

pub const MAGIC: &[u8; 8] = b"SOV_ARCA";
pub const ALIGN: usize = 64;

#[derive(Debug)]
pub enum SovereignError {
    InvalidMagicBytes,
    HeaderParsingFailed,
    TensorNotFound(String),
    AlignmentError,
    IOError(std::io::Error),
    ShapeMismatch { name: String, expected: Vec<usize>, got: Vec<usize> },
}

impl From<std::io::Error> for SovereignError {
    fn from(e: std::io::Error) -> Self {
        SovereignError::IOError(e)
    }
}

impl std::fmt::Display for SovereignError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SovereignError::InvalidMagicBytes => write!(f, "Invalid magic bytes: expected SOV_ARCA"),
            SovereignError::HeaderParsingFailed => write!(f, "Failed to parse sovereign header JSON"),
            SovereignError::TensorNotFound(n) => write!(f, "Tensor not found in registry: {}", n),
            SovereignError::AlignmentError => write!(f, "Tensor data is not 64-byte aligned"),
            SovereignError::IOError(e) => write!(f, "IO error: {}", e),
            SovereignError::ShapeMismatch { name, expected, got } => {
                write!(f, "Shape mismatch for '{}': expected {:?}, got {:?}", name, expected, got)
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SovereignHeader {
    pub num_layers: usize,
    pub d_model: usize,
    pub n_res: usize,
    pub lsh_k: usize,
    pub rank_r: usize,
    pub lsm_seed: u64,
    pub lsm_density: f32,
    pub vocab_size: usize,
    pub top_k_out: usize,
}

impl Default for SovereignHeader {
    fn default() -> Self {
        SovereignHeader {
            num_layers: 4,
            d_model: 512,
            n_res: 4096,
            lsh_k: 32,
            rank_r: 32,
            lsm_seed: 0xDEAD_BEEF_CAFE_1234,
            lsm_density: 0.05,
            vocab_size: 50_000,
            top_k_out: 200,
        }
    }
}

/// Registry entry: (byte_offset_in_mmap, byte_length, shape)
pub type TensorEntry = (usize, usize, Vec<usize>);

pub struct SovereignModel {
    pub header: SovereignHeader,
    mmap: Mmap,
    pub tensor_registry: HashMap<String, TensorEntry>,
}

impl SovereignModel {
    // ------------------------------------------------------------------ load
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, SovereignError> {
        let file = File::open(path).map_err(SovereignError::IOError)?;
        let mmap = unsafe { Mmap::map(&file).map_err(SovereignError::IOError)? };

        if &mmap[0..8] != MAGIC {
            return Err(SovereignError::InvalidMagicBytes);
        }

        let header_len = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let header_json = std::str::from_utf8(&mmap[16..16 + header_len])
            .map_err(|_| SovereignError::HeaderParsingFailed)?;
        let header: SovereignHeader = serde_json::from_str(header_json)
            .map_err(|_| SovereignError::HeaderParsingFailed)?;

        // advance past header + padding to 64-byte boundary
        let raw_after_header = 16 + header_len;
        let dict_start = align_up(raw_after_header, ALIGN);

        let tensor_registry = Self::parse_tensor_dict(&mmap, dict_start)?;

        Ok(SovereignModel { header, mmap, tensor_registry })
    }

    fn parse_tensor_dict(
        mmap: &[u8],
        mut cursor: usize,
    ) -> Result<HashMap<String, TensorEntry>, SovereignError> {
        let mut registry = HashMap::new();

        // Number of entries stored as u32 LE at dict_start
        if cursor + 4 > mmap.len() {
            return Ok(registry); // empty model file (only header)
        }
        let n_entries = u32::from_le_bytes(mmap[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;

        for _ in 0..n_entries {
            // name_len (u32)
            let name_len = u32::from_le_bytes(mmap[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            // name bytes
            let name = std::str::from_utf8(&mmap[cursor..cursor + name_len])
                .map_err(|_| SovereignError::HeaderParsingFailed)?
                .to_string();
            cursor += name_len;
            // offset (u64), length (u64)
            let offset = u64::from_le_bytes(mmap[cursor..cursor + 8].try_into().unwrap()) as usize;
            cursor += 8;
            let byte_len = u64::from_le_bytes(mmap[cursor..cursor + 8].try_into().unwrap()) as usize;
            cursor += 8;
            // shape_len (u32) then shape dims (u32[])
            let shape_len = u32::from_le_bytes(mmap[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                let dim = u32::from_le_bytes(mmap[cursor..cursor + 4].try_into().unwrap()) as usize;
                cursor += 4;
                shape.push(dim);
            }
            registry.insert(name, (offset, byte_len, shape));
        }
        Ok(registry)
    }

    // ------------------------------------------------------------------ tensor views
    /// Zero-copy view into the mmap as a flat f32 slice.
    pub fn tensor_f32_slice(&self, name: &str) -> Result<&[f32], SovereignError> {
        let (offset, byte_len, _shape) = self
            .tensor_registry
            .get(name)
            .ok_or_else(|| SovereignError::TensorNotFound(name.to_string()))?;

        if offset % 4 != 0 {
            return Err(SovereignError::AlignmentError);
        }
        let ptr = self.mmap[*offset..*offset + byte_len].as_ptr() as *const f32;
        let len = byte_len / 4;
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    pub fn tensor_as_array2(&self, name: &str) -> Result<Array2<f32>, SovereignError> {
        let (offset, byte_len, shape) = self
            .tensor_registry
            .get(name)
            .ok_or_else(|| SovereignError::TensorNotFound(name.to_string()))?;

        assert_eq!(shape.len(), 2, "tensor_as_array2 requires rank-2 shape");
        let (rows, cols) = (shape[0], shape[1]);
        assert_eq!(rows * cols * 4, *byte_len);

        let slice = self.tensor_f32_slice(name)?;
        Ok(Array2::from_shape_vec((rows, cols), slice.to_vec()).unwrap())
    }

    pub fn tensor_as_array1(&self, name: &str) -> Result<Array1<f32>, SovereignError> {
        let (_offset, byte_len, shape) = self
            .tensor_registry
            .get(name)
            .ok_or_else(|| SovereignError::TensorNotFound(name.to_string()))?;
        let n = byte_len / 4;
        let slice = self.tensor_f32_slice(name)?;
        Ok(Array1::from_vec(slice.to_vec()))
    }

    // ------------------------------------------------------------------ LSM generation
    /// Reconstruct sparse R matrix on-the-fly from seed. Never stored on disk.
    pub fn generate_sparse_lsm(&self) -> Array2<f32> {
        let n = self.header.n_res;
        let density = self.header.lsm_density as f64;
        let std_dev = (1.0_f32 / n as f32).sqrt();

        let mut rng = Pcg64::seed_from_u64(self.header.lsm_seed);
        let normal = Normal::new(0.0_f32, std_dev).unwrap();

        let mut matrix = Array2::<f32>::zeros((n, n));
        // We need a uniform [0,1) sampler from the same rng stream.
        use rand::Rng;
        for i in 0..n {
            for j in 0..n {
                if rng.gen_bool(density) {
                    matrix[[i, j]] = normal.sample(&mut rng);
                }
            }
        }
        matrix
    }

    // ------------------------------------------------------------------ save
    pub fn save_to_file<P: AsRef<Path>>(
        header: &SovereignHeader,
        tensors: &[(&str, &[f32], &[usize])],
        path: P,
    ) -> Result<(), SovereignError> {
        let mut buf: Vec<u8> = Vec::new();

        // Magic
        buf.extend_from_slice(MAGIC);

        // Header JSON placeholder (8 bytes for length)
        let header_json = serde_json::to_string(header).unwrap();
        let h_bytes = header_json.as_bytes();
        let h_len = h_bytes.len() as u64;
        buf.extend_from_slice(&h_len.to_le_bytes());
        buf.extend_from_slice(h_bytes);

        // Pad to 64-byte boundary
        while buf.len() % ALIGN != 0 {
            buf.push(0x00);
        }

        // ---- Build dict bytes first to know offsets ----
        // dict_start = current buf.len()
        // dict: u32 n_entries | [ name_len(u32) | name | offset(u64) | byte_len(u64) | shape_len(u32) | shape(u32[]) ]*
        let n_entries = tensors.len() as u32;
        let mut dict_buf: Vec<u8> = Vec::new();
        dict_buf.extend_from_slice(&n_entries.to_le_bytes());

        // First pass: compute dict byte size to know where data payload starts
        let dict_entry_size: usize = tensors
            .iter()
            .map(|(name, _data, shape)| {
                4 + name.len() + 8 + 8 + 4 + shape.len() * 4
            })
            .sum();
        let dict_total = 4 + dict_entry_size; // includes the n_entries u32

        // Data payload starts after dict, aligned to 64 bytes
        let dict_end_abs = buf.len() + dict_total;
        let data_start_abs = align_up(dict_end_abs, ALIGN);

        // Second pass: fill dict with correct offsets
        let mut data_cursor = data_start_abs;
        for (name, data, shape) in tensors.iter() {
            let name_bytes = name.as_bytes();
            let byte_len = data.len() * 4;

            dict_buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            dict_buf.extend_from_slice(name_bytes);
            dict_buf.extend_from_slice(&(data_cursor as u64).to_le_bytes());
            dict_buf.extend_from_slice(&(byte_len as u64).to_le_bytes());
            dict_buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &dim in *shape {
                dict_buf.extend_from_slice(&(dim as u32).to_le_bytes());
            }
            data_cursor += align_up(byte_len, ALIGN);
        }

        buf.extend_from_slice(&dict_buf);

        // Pad to data payload start
        while buf.len() < data_start_abs {
            buf.push(0x00);
        }

        // Data payloads
        for (_name, data, _shape) in tensors.iter() {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            };
            buf.extend_from_slice(bytes);
            while buf.len() % ALIGN != 0 {
                buf.push(0x00);
            }
        }

        std::fs::write(path, &buf).map_err(SovereignError::IOError)
    }
}

#[inline(always)]
pub fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}
