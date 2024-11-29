from llama_zip import LlamaZip

# Initialize the compressor
compressor = LlamaZip(model_path="unsloth/Llama-3.2-1B")

# Compress some data
original = b"The quick brown fox jumps over the lazy dog."
compressed = compressor.compress(original)
assert len(compressed) < len(original)

# Decompress the data
decompressed = compressor.decompress(compressed)
assert decompressed == original
