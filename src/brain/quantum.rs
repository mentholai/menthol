use super::types::ThoughtVector;
use crate::models::{ComputeDevice, Result};
use rand::Rng;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct QuantumProcessor {
    state: Arc<RwLock<QuantumState>>,
    device: ComputeDevice,
    coherence_threshold: f64,
}

pub struct QuantumState {
    pub entanglement_matrix: Vec<Vec<f64>>,
    pub superposition_vector: Vec<f64>,
    pub collapse_probability: f64,
}

impl QuantumProcessor {
    pub fn new_with_device(device: ComputeDevice) -> Self {
        Self {
            state: Arc::new(RwLock::new(Self::initialize_quantum_state())),
            device,
            coherence_threshold: 0.7,
        }
    }

    fn initialize_quantum_state() -> QuantumState {
        let dimension = 64;
        let mut rng = rand::thread_rng();

        QuantumState {
            entanglement_matrix: (0..dimension)
                .map(|_| (0..dimension).map(|_| rng.gen::<f64>()).collect())
                .collect(),
            superposition_vector: (0..dimension)
                .map(|_| rng.gen::<f64>() * 2.0 - 1.0)
                .collect(),
            collapse_probability: 0.5,
        }
    }

    pub async fn transform_vector(&mut self, vector: &mut ThoughtVector) -> Result<()> {
        let mut rng = rand::thread_rng();
        let dimension = vector.len();

        // Apply quantum entanglement
        let mut entangled = vec![0.0; dimension];
        for i in 0..dimension {
            for j in 0..dimension {
                entangled[i] += vector[j] * rng.gen::<f64>();
            }
        }

        // Apply superposition
        for i in 0..dimension {
            vector[i] = (vector[i] + entangled[i]) * 0.5 + (rng.gen::<f64>() * 2.0 - 1.0) * 0.1;
        }

        // Random quantum collapse
        if rng.gen::<f64>() < 0.3 {
            self.quantum_collapse(vector)?;
        }

        Ok(())
    }

    fn quantum_collapse(&self, vector: &mut ThoughtVector) -> Result<()> {
        let mut rng = rand::thread_rng();
        let collapse_point = rng.gen_range(0..vector.len());

        // Collapse around a random point with exponential decay
        let value = vector[collapse_point];
        for i in 0..vector.len() {
            let distance = (i as f64 - collapse_point as f64).abs();
            let collapse_factor = (-distance * 0.1).exp();
            vector[i] = vector[i] * (1.0 - collapse_factor) + value * collapse_factor;
        }

        Ok(())
    }

    pub fn measure_coherence(&self, vector: &ThoughtVector) -> f64 {
        let mut coherence = 0.0;
        let len = vector.len();

        for i in 0..len {
            for j in 0..len {
                coherence += vector[i] * vector[j];
            }
        }

        (coherence / (len * len) as f64).abs()
    }

    pub fn get_device(&self) -> &ComputeDevice {
        &self.device
    }

    pub fn get_coherence_threshold(&self) -> f64 {
        self.coherence_threshold
    }
}

///NOTE: Quantum functions are strictly for fun, you didnt really think your computer could handle Quantum computing did you?  

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherence_measurement() {
        let processor = QuantumProcessor::new_with_device(ComputeDevice::CPU);
        let vector = vec![1.0, 1.0, 1.0, 1.0];
        let coherence = processor.measure_coherence(&vector);
        assert!(coherence >= 0.0 && coherence <= 1.0);
    }
}
