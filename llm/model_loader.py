"""LLM Model Loader — wraps a local GGUF model with a simple generate(prompt) interface."""

import yaml
from pathlib import Path

# Global model instance
_model = None
_config = None


def _load_config():
    """Load LLM configuration from config.yaml."""
    global _config
    if _config is None:
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r") as f:
            _config = yaml.safe_load(f)
    return _config


class ModelLoader:
    """Manages loading and inference of a local LLM (GGUF format via llama-cpp-python)."""

    def __init__(self, model_path: str = None):
        config = _load_config()
        self.model_path = model_path or config.get("model_path", "models/llm/model.gguf")
        self.n_ctx = config.get("n_ctx", 2048)
        self.n_threads = config.get("n_threads", 4)
        self.max_tokens = config.get("max_tokens", 256)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.model = None

    def load(self):
        """Load the model into memory."""
        try:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
            print(f"[LLM] Model loaded from {self.model_path}")
        except Exception as e:
            print(f"[LLM] Error loading model: {e}")
            raise

    def generate_response(self, prompt: str) -> str:
        """Generate a response from the loaded model."""
        if self.model is None:
            self.load()

        output = self.model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=["\nUser:", "\nHuman:"],
        )
        return output["choices"][0]["text"].strip()

    def unload(self):
        """Unload the model from memory."""
        self.model = None
        print("[LLM] Model unloaded.")


def generate(prompt: str) -> str:
    """Convenience function — uses a singleton ModelLoader."""
    global _model
    if _model is None:
        _model = ModelLoader()
        _model.load()
    return _model.generate_response(prompt)
