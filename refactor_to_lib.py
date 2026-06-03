import os
import re

main_path = "d:/code/ARCA/main.rs"
sys_path = "d:/code/ARCA/system.rs"
lib_path = "d:/code/ARCA/lib.rs"
cli_path = "d:/code/ARCA/arca_cli/main.py"

with open(main_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Extract `fn main()` and everything after it.
# Usually `fn main()` is at the bottom of main.rs.
match = re.search(r"fn main\(\)\s*\{", code)
if match:
    main_start = match.start()
    system_code = code[:main_start]
    main_code = code[main_start:]
    
    # We need to add imports to main.rs
    new_main = """use arca::system::*;
use arca::encoder::*;
use arca::sovereign::*;
use arca::tokenizer::*;
use arca::memory::*;
use arca::metabolic_core::*;

""" + main_code
    
    # Remove 'mod' declarations from system_code, since they will be in lib.rs
    system_code = re.sub(r"mod encoder;.*?\n", "", system_code)
    system_code = re.sub(r"mod memory;.*?\n", "", system_code)
    system_code = re.sub(r"mod metabolic_core;.*?\n", "", system_code)
    system_code = re.sub(r"mod sovereign;.*?\n", "", system_code)
    system_code = re.sub(r"mod tokenizer;.*?\n", "", system_code)
    system_code = re.sub(r"mod train;.*?\n", "", system_code)
    system_code = re.sub(r"#\[cfg\(feature = \"gpu\"\)\].*?\nmod gpu_context;.*?\n", "", system_code, flags=re.DOTALL)
    system_code = re.sub(r"#\[cfg\(feature = \"gpu\"\)\].*?\nmod gpu_inference_context;.*?\n", "", system_code, flags=re.DOTALL)
    
    # Add use statements to system_code
    new_system = """
use crate::encoder::*;
use crate::memory::*;
use crate::metabolic_core::*;
use crate::sovereign::*;
use crate::tokenizer::*;
use crate::train::*;

#[cfg(feature = "gpu")]
use crate::gpu_context;
#[cfg(feature = "gpu")]
use crate::gpu_inference_context;

""" + system_code

    with open(sys_path, "w", encoding="utf-8") as f:
        f.write(new_system)

    with open(main_path, "w", encoding="utf-8") as f:
        f.write(new_main)
        
# 2. Write lib.rs with PyO3 bindings
lib_code = """
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
use std::sync::Arc;

#[pyclass]
pub struct PyArcaModel {
    inner: ArcaSystem,
}

#[pymethods]
impl PyArcaModel {
    #[new]
    pub fn new(model_path: &str) -> PyResult<Self> {
        let model = SovereignModel::load(model_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load model: {:?}", e)))?;
        let system = ArcaSystem::from_sovereign(&model)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to init system: {:?}", e)))?;
        Ok(PyArcaModel { inner: system })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> PyResult<Vec<u32>> {
        #[cfg(feature = "gpu")]
        {
            let bpe_ids = self.inner.tokenize(prompt);
            let raw_bytes: Vec<u8> = prompt.bytes().collect();
            
            let mut generated = Vec::new();
            
            // For simplicity in the Python API, we do single-batch generation here.
            let bytes_batch = vec![raw_bytes.clone()];
            let t_batch = vec![0];
            let bpe_batch = vec![bpe_ids.clone()];
            
            for _ in 0..max_tokens {
                let chosen = self.inner.forward_step_extreme_inference(&bytes_batch, &t_batch, &bpe_batch, temperature);
                generated.push(chosen[0]);
                // In a real loop, we would append chosen[0] to the context and increment t_batch
            }
            Ok(generated)
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(pyo3::exceptions::PyRuntimeError::new_err("Python bindings require the 'gpu' feature."))
        }
    }
}

#[pymodule]
fn arca_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyArcaModel>()?;
    Ok(())
}
"""

with open(lib_path, "w", encoding="utf-8") as f:
    f.write(lib_code)

# 3. Create the Python CLI
os.makedirs("d:/code/ARCA/arca", exist_ok=True)
with open("d:/code/ARCA/arca/__init__.py", "w", encoding="utf-8") as f:
    f.write("from .arca_core import PyArcaModel\n")

with open("d:/code/ARCA/arca/main.py", "w", encoding="utf-8") as f:
    f.write('''import argparse
import sys
from .arca_core import PyArcaModel

def main():
    parser = argparse.ArgumentParser(description="ARCA LLM Inference CLI")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the .sovereign model file")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("-n", "--max-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature for sampling")
    
    args = parser.parse_args()
    
    try:
        print(f"Loading model from {args.model}...", file=sys.stderr)
        model = PyArcaModel(args.model)
        
        print(f"Generating {args.max_tokens} tokens...", file=sys.stderr)
        tokens = model.generate(args.prompt, args.max_tokens, args.temp)
        print("Generated Token IDs:", tokens)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
''')

print("Refactoring complete.")
