# Development Guide for Stem+MIDI Pro

This guide provides information for developers who want to contribute to the Stem+MIDI Pro project, understand the codebase, or extend its functionality.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Codebase Overview](#codebase-overview)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Extending the Model](#extending-the-model)
7. [Adding New Features](#adding-new-features)
8. [Documentation Standards](#documentation-standards)
9. [Code Style](#code-style)
10. [Contributing](#contributing)

## Getting Started

Before you begin, ensure you have the following installed:
- Python 3.8+
- Git
- (Optional) Docker for containerized development

Clone the repository:
```bash
git clone <repository-url>
cd stem_midi_pro
```

## Codebase Overview

### Core Components
- `main.py`: NeMo ModelPT wrapper - the main entry point for the model
- `api.py`: FastAPI application providing RESTful endpoints
- `models/`: Contains the neural network components
  - `mamba_separator.py`: Source separation using Mamba-SSM
  - `mamba_transcriber.py`: Audio-to-MIDI transcription with expression detection
  - `confidence_injector.py`: Adds metadata to MIDI events
  - `losses.py`: Perceptual loss functions
- `utils/`: Utility functions
  - `quality_gates.py`: Confidence-based routing logic
- `data/`: Data loading and preprocessing
  - `canadian_datasets.py`: Canadian artist dataset loaders
- `user_content/`: All user-facing text and prompts
- `configs/`: Configuration files
  - `model_config.yaml`: Model architecture and training parameters

### Data Flow
1. Audio input → Validation → Preprocessing
2. Separation module → Guitar/bass stems
3. Transcription module → MIDI events + confidence scores
4. Confidence injection → Metadata enrichment
5. Quality gates → Routing decision (studio/draft/complex)
6. Output formatting → ZIP file with stems, MIDI, report

## Development Setup

### Prerequisites
- Python 3.8+
- pip
- Git
- CUDA-capable GPU (for training, optional for development)
- Conda or virtualenv (recommended)

### Setup Steps
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install development dependencies (optional but recommended):
   ```bash
   pip install pytest black flake8 isort mypy
   ```

4. Pre-commit hooks (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Configuration for Development
- Use the synthetic data generators in `data/canadian_datasets.py` for initial development
- Modify `example_data_config.yaml` to point to your local data
- Set environment variables if needed:
  ```bash
  export MODEL_CONFIG_PATH=configs/model_config.yaml
  export MODEL_CHECKPOINT_PATH=path/to/your/checkpoint.nemo  # Optional
  ```

## Making Changes

### Best Practices
1. Always create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Write clear, concise commit messages:
   ```
   feat: add new expression detection for tremolo
   fix: resolve clipping artifact in separation module
   docs: update API documentation for new endpoint
   ```

3. Follow the existing code style (see [Code Style](#code-style))

4. Update documentation when changing user-facing behavior

5. Add tests for new functionality

### Working with Models
When modifying the neural network components:
1. Understand the input/output shapes documented in each class
2. Maintain backward compatibility when possible
3. Update the model configuration if adding new hyperparameters
4. Test with synthetic data before using real datasets
5. Consider the impact on training time and memory usage

### API Changes
When modifying the API:
1. Keep backward compatibility for existing endpoints
2. Update the API documentation (`API_DOCUMENTATION.md`)
3. Consider versioning if making breaking changes
4. Test all endpoints with the provided examples

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=stem_midi_pro

# Run specific test file
pytest tests/test_specific_feature.py
```

### Writing Tests
- Place tests in a `tests/` directory (create if needed)
- Use pytest fixtures for common setup
- Mock external dependencies when possible
- Test both success and failure cases
- For model tests, use synthetic data to ensure reproducibility

### Test Coverage Goals
- Aim for >80% coverage on new code
- Critical paths (audio processing, quality gates) should have near-complete coverage
- API endpoints should be tested for status codes and response formats

## Extending the Model

### Adding New Output Stems
To separate additional instruments (e.g., vocals, drums):
1. Modify `mamba_separator.py`:
   - Add new mask head in `__init__`
   - Update the mask generation loop in `forward`
   - Update the stem dictionary creation
   - Update output types in `input_types`/`output_types` if using NeMo typing
2. Update the model configuration to specify new stem types
3. Modify the processing pipeline in `main.py` to handle the new stem
4. Update the API to return the new stem in the ZIP file
5. Update user-facing content to mention the new capability

### Adding New Expression Types
To detect additional musical expressions (e.g., tremolo, wah):
1. Modify `mamba_transcriber.py`:
   - Add new expression type to `expression_heads` list
   - Update the expression head output dimension
   - Update the expression processing in `forward`
2. Update the confidence injector to handle the new expression type
3. Update the MIDI generation to include the new expression data
4. Update the user interface to display/edit the new expression
5. Update training data to include examples of the new expression

### Changing Model Architecture
To modify the Mamba architecture (layers, dimensions, etc.):
1. Update `configs/model_config.yaml` with new hyperparameters
2. The model classes read from the config, so most changes will be automatic
3. For structural changes (e.g., adding skip connections), modify the model classes directly
4. Always test that the model still loads and runs inference correctly

## Adding New Features

### New Processing Stages
To add a new stage to the processing pipeline:
1. Implement the stage as a separate function or class
2. Integrate it into the `forward` method of `StemMidiModel` in `main.py`
3. Update the processing report if it generates new metrics
4. Update the quality gates if it affects routing decisions
5. Update the API response to include any new outputs
6. Add user-facing documentation for the new feature

### New Output Formats
To support additional output formats (e.g., FLAC stems, MusicXML):
1. Modify the output packaging in `api.py` (in `create_response_zip` or similar)
2. Add the necessary conversion functions (may require additional libraries)
3. Update the file extensions in the ZIP file
4. Update the API documentation to mention the new format
5. Consider adding a format selection parameter to the API endpoint

### Advanced Routing Logic
To enhance the quality-based routing:
1. Modify `utils/quality_gates.py` to add new quality tiers or conditions
2. Update the routing decision structure if needed
3. Update the user-facing messages in `user_content/`
4. Consider adding user-configurable thresholds via API parameters
5. Update the documentation to explain the new routing options

## Documentation Standards

### Docstrings
- Use Google-style docstrings for all classes and methods
- Include Args, Returns, and Raises sections where applicable
- Document tensor shapes using (B, C, T) notation for batch, channels, time
- Example:
  ```python
  def forward(self, audio: torch.Tensor) -> torch.Tensor:
      """Process audio through the separation network.

      Args:
          audio: Input audio tensor of shape (B, C, T) where
                 B = batch size, C = number of channels (usually 1),
                 T = number of time samples.

      Returns:
          Tuple of (guitar_stem, bass_stem, residual, new_state, metrics)
          where each stem has shape (B, C, T) and metrics has shape (B, 2)
          containing [SI-SDR estimate, phase coherence].
      """
  ```

### Markdown Files
- Use `#` for main title, `##` for sections, `###` for subsections
- Keep line length under 100 characters for readability
- Use fenced code blocks with language specifiers for code snippets
- Use tables for structured data
- Link to other documents using relative paths
- Include a table of contents for long documents

### API Documentation
- Follow the OpenAPI/Swagger specification (FastAPI generates this automatically)
- Keep descriptions concise but informative
- Include example requests and responses
- Document all possible response codes
- Mention any authentication or rate limiting requirements

## Code Style

### Python
- Follow PEP 8 as much as possible
- Use 4 spaces for indentation (no tabs)
- Limit lines to 79 characters
- Use descriptive variable and function names
- Prefer `is None`/`is not None` over `== None`
- Use list comprehensions where appropriate
- Import order: standard library, third-party, local imports
- Use type hints for function signatures and class attributes
- Use `__all__` to control what is exported from modules

### YAML Configuration
- Use 2 spaces for indentation
- Use descriptive keys in snake_case
- Group related parameters under common headings
- Comment non-obvious values
- Use quotes for strings that could be misinterpreted (e.g., "yes", "no")

### Shell Scripts
- Use `#!/usr/bin/env bash` for portability
- Set `-euo pipefail` for safer scripting
- Quote variable expansions
- Use meaningful variable names
- Comment complex logic

## Contributing

### Reporting Issues
- Use the GitHub issue tracker
- Include steps to reproduce for bug reports
- Include expected vs actual behavior
- Include relevant logs or error messages
- For feature requests, describe the use case and benefits

### Submitting Changes
1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Ensure tests pass
5. Update documentation
6. Submit a pull request with a clear description
7. Link to any relevant issues
8. Wait for code review and address feedback

### Code Review Process
- Look for correctness, clarity, and adherence to standards
- Check for potential performance issues
- Ensure security considerations are addressed
- Verify that tests cover the new functionality
- Confirm documentation is updated
- Be respectful and constructive in feedback

### Licensing
By contributing, you agree that your contributions will be licensed under the same license as the project.

## Useful Commands

### Development
```bash
# Run the demo
python demo.py

# Start the API server
uvicorn api:app --reload

# Run tests
pytest

# Check code style
flake8 stem_midi_pro

# Format code
black stem_midi_pro

# Sort imports
isort stem_midi_pro

# Type checking
mypy stem_midi_pro
```

### Data Preparation
```bash
# Generate synthetic data for testing
python -c "from data.canadian_datasets import CanadianAudioDataset; d = CanadianAudioDataset('./data/synthetic'); print(len(d))"

# Validate data configuration
python -c "import yaml; config = yaml.safe_load(open('example_data_config.yaml')); print('Config loaded')"
```

### Model Operations
```bash
# Load model and check summary
python -c "from main import StemMidiModel; import yaml; config = yaml.safe_load(open('configs/model_config.yaml')); model = StemMidiModel(config); print(model)"

# Export to TorchScript (for testing)
python -c "from main import StemMidiModel; import yaml; import torch; config = yaml.safe_load(open('configs/model_config.yaml')); model = StemMidiModel(config); model.eval(); example_input = torch.randn(1, 1, 44100*5); traced_model = torch.jit.trace(model, example_input); traced_model.save('model_traced.pt')"
```

---
*Development Guide Version: 1.0*
*Last Updated: $(date)*