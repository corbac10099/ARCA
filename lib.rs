
pub mod encoder;
pub mod memory;
pub mod metabolic_core;
pub mod sovereign;
pub mod tokenizer;
pub mod train;

#[cfg(feature = "gpu")]
pub mod gpu_context;
#[cfg(feature = "gpu")]
pub mod gpu_inference_context;

pub mod system;

use pyo3::prelude::*;
use system::ArcaSystem;
use sovereign::SovereignModel;
use tokenizer::BpeTokenizer;
use std::sync::Arc;

#[pyclass]
pub struct PyArcaModel {
    inner: ArcaSystem,
    tokenizer: BpeTokenizer,
}

#[pymethods]
impl PyArcaModel {
    #[new]
    pub fn new(model_path: &str) -> PyResult<Self> {
        let model = SovereignModel::load(model_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load model: {:?}", e)))?;
        let system = ArcaSystem::from_sovereign(&model)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to init system: {:?}", e)))?;
        let tokenizer = BpeTokenizer::new_base(); // Default fallback
        Ok(PyArcaModel { inner: system, tokenizer })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> PyResult<Vec<u32>> {
        #[cfg(feature = "gpu")]
        {
            let raw_bytes: Vec<u8> = prompt.bytes().collect();
            let bpe_ids = self.tokenizer.encode(&raw_bytes);
            
            let mut generated = Vec::new();
            
            // For simplicity in the Python API, we do single-batch generation here.
            let bytes_batch = vec![raw_bytes.clone()];
            let mut t_batch = vec![0];
            let bpe_batch = vec![bpe_ids.clone()];
            
            for _ in 0..max_tokens {
                let chosen = self.inner.forward_step_extreme_inference(&bytes_batch, &t_batch, &bpe_batch, temperature);
                generated.push(chosen[0]);
                t_batch[0] += 1;
            }
            Ok(generated)
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(pyo3::exceptions::PyRuntimeError::new_err("Python bindings require the 'gpu' feature."))
        }
    }

    pub fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        let bytes = self.tokenizer.decode(&ids);
        String::from_utf8(bytes).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid UTF-8: {:?}", e)))
    }
}

#[pymodule]
fn arca_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyArcaModel>()?;
    Ok(())
}
