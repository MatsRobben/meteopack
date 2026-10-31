"""meteopack — ingest, canonicalise and pack multi-source meteorological data.

Three layers, defined in docs/architecture: raw (immutable, as fetched),
canonical (one store per source/product, CF names and units), derived
(training-shaped bricks and station matrices). Raw is never read by a model.
"""

__version__ = "0.1.0"
