# Vendored from HKUDS/RAG-Anything @ 32eef6e; MIT license, see raganything/LICENSE.
from enum import Enum


class DocStatus(str, Enum):
    """Document processing status"""

    READY = "ready"
    HANDLING = "handling"
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
