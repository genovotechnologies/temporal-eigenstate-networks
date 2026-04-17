from .models.ten_unified import TEN
from .models.ten_fft import TENFastModel, TENFastLayer, eigenstate_fft
from .models.ten_chunked import TENChunkedModel, TENChunkedLayer
from .models.ten_pro import TENProModel, TENProLayer

__version__ = "0.1.0"
__all__ = [
    "TEN",  # The main export — auto-selects FFT or Pro based on context length
    "TENFastModel", "TENFastLayer", "eigenstate_fft",
    "TENChunkedModel", "TENChunkedLayer",
    "TENProModel", "TENProLayer",
]
