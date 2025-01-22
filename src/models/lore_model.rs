use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use std::path::PathBuf;

pub struct LoreGenerator {
    model: LlamaModel,
}

impl LoreGenerator {
    /// **Initialize the LLaMA model**
    pub fn new(model_path: PathBuf) -> Self {
        let model = LlamaModel::load_from_file(&model_path, LlamaParams::default())
            .expect("Could not load LLaMA model");

        Self { model }
    }

    /// **Generate a name for the character**
    pub fn generate_name(&self, description: &str) -> String {
        let prompt = format!(
            "Give a fantasy name for this character:\n{}\n\nCharacter Name: ",
            description
        );

        let mut session = self
            .model
            .create_session(SessionParams::default())
            .expect("Failed to create session");

        session.advance_context(&prompt).unwrap();

        let completions = session
            .start_completing_with(
                llama_cpp::standard_sampler::StandardSampler::default(),
                10, // Name should be short
            )
            .expect("Failed to start completion")
            .into_strings();

        let mut name = String::new();
        for completion in completions {
            name.push_str(&completion);
        }

        name.trim().to_string()
    }

    /// **Generate a full backstory/lore for the character**
    pub fn generate_lore(&self, description: &str) -> String {
        let prompt = format!(
            "Describe this character in a fantasy setting:\n{}\n\nFull backstory: ",
            description
        );

        let mut session = self
            .model
            .create_session(SessionParams::default())
            .expect("Failed to create session");

        session.advance_context(&prompt).unwrap();

        let completions = session
            .start_completing_with(llama_cpp::standard_sampler::StandardSampler::default(), 256)
            .expect("Failed to start completion")
            .into_strings();

        let mut response = String::new();
        for completion in completions {
            response.push_str(&completion);
        }

        response.trim().to_string()
    }
}

fn main() {
    let model_path = PathBuf::from("models/mistral.gguf");
    let lore_generator = LoreGenerator::new(model_path);

    let character_desc = "A mystical elven warrior with silver hair and glowing runes.";

    let name = lore_generator.generate_name(character_desc);
    let lore = lore_generator.generate_lore(character_desc);

    println!("Character Name: {}\n", name);
    println!("Generated Lore:\n{}", lore);
}
