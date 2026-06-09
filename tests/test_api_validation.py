"""Tests for API audio validation."""
import pytest
import tempfile
import numpy as np
import soundfile as sf

@pytest.mark.skip(reason="requires mamba_ssm (CUDA) — api.py imports StemMidiModel at module level")
def test_validate_audio_valid_file():
    """Valid WAV file should pass validation."""
    from api import validate_audio_file
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, mode='wb') as f:
        dummy = np.random.randn(44100 * 5).astype(np.float32)
        sf.write(f.name, dummy, 44100)
        info = validate_audio_file(f.name)
    
    assert info['duration'] == pytest.approx(5.0, rel=0.1)
    assert info['sample_rate'] == 44100

@pytest.mark.skip(reason="requires mamba_ssm (CUDA) — api.py imports StemMidiModel at module level")
def test_validate_audio_too_long():
    """Audio exceeding MAX_DURATION should raise HTTPException."""
    from api import validate_audio_file
    from fastapi import HTTPException
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, mode='wb') as f:
        dummy = np.random.randn(44100 * 700).astype(np.float32)
        sf.write(f.name, dummy, 44100)
        with pytest.raises(HTTPException):
            validate_audio_file(f.name)

@pytest.mark.skip(reason="requires mamba_ssm (CUDA) — api.py imports StemMidiModel at module level")
def test_validate_audio_unsupported_sr():
    """Unsupported sample rates should raise HTTPException."""
    from api import validate_audio_file
    from fastapi import HTTPException
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, mode='wb') as f:
        dummy = np.random.randn(22050 * 5).astype(np.float32)
        sf.write(f.name, dummy, 22050)
        with pytest.raises(HTTPException):
            validate_audio_file(f.name)
