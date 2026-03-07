# Biometric Scanner

Scans phone dumps for biometric data files (fingerprint, facial recognition, gait, iris, voice).

## Usage

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

## Options

- `--recursive, -r` - Full recursive scan (default: top-level only)
- `--output, -o` - Output JSON file (default: stdout)
- `--custom-patterns, -c` - JSON file with custom detection patterns
- `--parallel, -p` - Number of parallel workers (default: 4)
- `--no-extract` - Disable data extraction from files

## Custom Patterns

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

## Output

JSON with file paths, biometric types, confidence levels, detection methods, and extracted data.
