use image::io::Reader as ImageReader;
use image::DynamicImage;
use ndarray::{ArrayD, CowArray, IxDyn};
use ort::{Environment, ExecutionProvider, Session, SessionBuilder, Value};
use std::path::PathBuf;
use std::sync::Arc;

pub struct ImageAnalyzer {
    model: Session, // ONNX runtime session for running the model
}

impl ImageAnalyzer {
    /// Creates a new `ImageAnalyzer` instance with the given ONNX model path.
    pub fn new(model_path: PathBuf) -> Self {
        //  Initialize the ONNX Runtime environment
        let env = Arc::new(
            Environment::builder()
                .with_name("clip") // Name the runtime environment
                .with_execution_providers([
                    ExecutionProvider::CUDA(Default::default()), // Use CUDA if available
                    ExecutionProvider::CPU(Default::default()),  // Fallback to CPU
                ])
                .build()
                .expect("Failed to create ONNX runtime environment"), // Panic if environment fails to initialize
        );

        // Load the ONNX model into the session
        let model = SessionBuilder::new(&env)
            .unwrap() // Ensure the session is created successfully
            .with_model_from_file(model_path) // Load the ONNX model file
            .unwrap(); // Panic if the model fails to load

        Self { model } // Return the ImageAnalyzer instance
    }

    pub fn analyze(&self, image_path: &PathBuf) -> String {
        // Load and decode the image
        let img = ImageReader::open(image_path)
            .unwrap() // Open the file
            .decode() // Convert it into a `DynamicImage`
            .unwrap(); // Ensure decoding is successful

        // Preprocess the image into a tensor format
        let tensor = self.preprocess_image(&img);

        // Convert the tensor into the required ONNX format (CowArray)
        let cow_tensor = CowArray::from(&tensor);

        // Prepare the ONNX model input
        let input = Value::from_array(self.model.allocator(), &cow_tensor)
            .expect("Failed to create input tensor");

        // Run the model inference
        let outputs = self
            .model
            .run(vec![input])
            .expect("ONNX model inference failed");

        // Extract the model's output tensor (features vector)
        let extracted_tensor: ort::tensor::OrtOwnedTensor<f32, IxDyn> = outputs[0]
            .try_extract()
            .expect("Failed to extract output tensor");

        // Convert extracted tensor data into a Rust vector of `f32`
        let features: Vec<f32> = extracted_tensor.view().iter().copied().collect();

        // Match extracted features to the most similar description
        self.match_features_to_text(&features)
    }

    /// Preprocesses an image to match the input format expected by the model.
    /// - Resizes the image to 224x224
    /// - Converts it to RGB format
    /// - Normalizes pixel values (0-255 → 0.0-1.0)
    fn preprocess_image(&self, img: &DynamicImage) -> ndarray::ArrayD<f32> {
        // Resize the image to the required input size (224x224)
        let img = img.resize_exact(224, 224, image::imageops::FilterType::Nearest);

        // Convert the image to RGB format
        let img = img.to_rgb8();

        // Convert the image into a NumPy-style 3D array (channels, height, width)
        let img_array = ndarray::Array3::<f32>::from_shape_fn((3, 224, 224), |(c, x, y)| {
            img.get_pixel(x as u32, y as u32)[c] as f32 / 255.0 // Normalize pixel values
        });

        img_array.into_dyn() // Convert to a dynamic-sized tensor
    }

    /// Matches extracted image features to a predefined set of descriptions using cosine similarity.
    /// This function simulates how CLIP would compare an image to known text embeddings.
    fn match_features_to_text(&self, features: &[f32]) -> String {
        //Define a set of possible character descriptions
        let descriptions = vec![
            (
                "A cybernetic warrior with glowing blue eyes",
                vec![0.2, 0.5, 0.8],
            ),
            (
                "A medieval knight clad in shining armor",
                vec![0.1, 0.7, 0.9],
            ),
            (
                "A mystical sorcerer wielding arcane power",
                vec![0.3, 0.6, 0.75],
            ),
        ];

        //Find the description with the highest similarity score
        let best_match = descriptions
            .iter()
            .max_by(|(_, emb1), (_, emb2)| {
                cosine_similarity(features, emb1)
                    .partial_cmp(&cosine_similarity(features, emb2))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(desc, _)| desc.clone()) // Extract the best matching text
            .unwrap_or_else(|| "An unknown character of mysterious origins");

        best_match.to_string()
    }
}

/// Computes the cosine similarity between two vectors (feature vectors).
/// Cosine similarity is commonly used in machine learning for comparing embeddings.
fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm1 * norm2)
}
