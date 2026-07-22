# Hansken Biometric Scanner Plugin

**v1.5.0** — Detects pre-computed biometric models and embedding caches during Hansken forensic extractions.

## How It Works

1. **Hansken creates file-level traces** from a disk image/archive (zip, UFDR, etc.)
2. **Plugin matcher** triggers on relevant file extensions (`.pkl`, `.pickle`, `.dat`, `.pb`, `.pt`, `.pth`, `.h5`, `.onnx`, `.npy`, `.npz`, `.json`, `.csv`, `.wav`, `.m4a`, `.mp3`)
3. **Plugin inspects** filename keywords and content (pickle opcodes, NPY/NPZ headers, JSON structure, CSV headers)
4. **Properties set** on the trace: `file.misc.biometricModelType`, `.framework`, `.detectedBy`, `.confidence`

## Build & Deploy

```bash
cd modules/biometric_scanner
tox                    # Run tests (7 tests)
tox -e package         # Build Docker image
# Deploy to Hansken and refresh:
curl http://localhost:9091/gatekeeper/tools?refresh=true
```

## Custom Patterns

Extend detection with `patterns.json` (mounted at `/app/patterns.json` in Docker):

```json
{
  "customType": {
    "extensions": [".xyz"],
    "patterns": ["keyword1", "keyword2"]
  }
}
```

## License

GNU General Public License v3.0 — [Lund University Hansken Research Group](https://github.com/Lund-University-Hansken-Research-Group)
