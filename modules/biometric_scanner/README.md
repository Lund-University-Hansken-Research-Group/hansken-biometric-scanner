# Biometric Scanner

Scans phone dumps for biometric data files (fingerprint, facial recognition, gait, iris, voice).

Also provides a **Hansken Extraction Plugin** to detect pre-computed biometric models during forensic extraction.

## Hansken Extraction Plugin

Detects pre-computed biometric models and embedding caches during Hansken extractions.

### Detected Types

| Type | Extensions | Description |
|------|------------|-------------|
| `faceRecognition` | `.pkl`, `.dat`, `.pb`, `.pt`, `.pth`, `.h5` | Face recognition models |
| `voiceBiometric` | `.wav`, `.m4a`, `.mp3`, `.pt`, `.pb` | Voice/speaker recognition models |
| `embeddingCache` | `.npy`, `.npz`, `.pkl`, `.json` | Pre-computed face embeddings |

### Trace Properties

The plugin sets the following properties on matching traces:

- `biometricModel.type` - Model type (faceRecognition, voiceBiometric, embeddingCache)
- `biometricModel.framework` - ML framework (dlib, tensorflow, pytorch, keras)
- `biometricModel.detectedBy` - Detection method (extension_pattern)
- `biometricModel.confidence` - Detection confidence (high, medium, low)

### Building

```bash
# Install Java 21 (for test framework)
source ~/.sdkman/bin/sdkman-init.sh
sdk install java 21.0.3-tem

# Run tests
tox

# Build Docker image
sudo docker build -t biometric-scanner-plugin .
```

### Deploying

```bash
# Push to Hansken registry
sudo docker tag biometric-scanner-plugin <registry>/prefix/biometric-scanner-plugin
sudo docker push <registry>/prefix/biometric-scanner-plugin

# Refresh Hansken tools
curl http://<hansken>/gatekeeper/tools?refresh=true
```

---

## CLI Scanner

Scans directories for biometric files using multiple detection methods.

### Usage

```bash
# Top-level scan (default)
python -m biometric_scanner.main scan /path/to/dump

# Full recursive scan
python -m biometric_scanner.main scan /path/to/dump --recursive

# Output to file
python -m biometric_scanner.main scan /path/to/dump -o results.json

# Custom patterns
python -m biometric_scanner.main scan /path/to/dump -c custom_patterns.json

# List supported biometric types
python -m biometric_scanner.main --list-types
```

### Options

- `--recursive, -r` - Full recursive scan (default: top-level only)
- `--output, -o` - Output JSON file (default: stdout)
- `--custom-patterns, -c` - JSON file with custom detection patterns
- `--parallel, -p` - Number of parallel workers (default: 4)
- `--no-extract` - Disable data extraction from files

### Custom Patterns

Create a JSON file:
```json
{
  "heartbeat": {
    "patterns": ["heartbeat", "heart_rate"],
    "extensions": [".hrb"],
    "confidence": "low"
  }
}
```

### Output

JSON with file paths, biometric types, confidence levels, detection methods, and extracted data.
