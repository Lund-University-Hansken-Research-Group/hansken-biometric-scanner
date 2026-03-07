#!/usr/bin/env python3
import argparse
import sys
import json
from pathlib import Path

from scanner.config import BiometricConfig
from scanner.scanner import BiometricScanner
from output.json_formatter import JsonFormatter, ScanInfo, BiometricMatch


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan phone dumps for biometric data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scan /path/to/dump
  %(prog)s scan /path/to/dump --recursive
  %(prog)s scan /path/to/dump --output results.json
  %(prog)s scan /path/to/dump --custom-patterns custom.json
  %(prog)s scan /path/to/dump --parallel 8
        """
    )
    
    parser.add_argument("--list-types", action="store_true",
                        help="List supported biometric types and exit")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    scan_parser = subparsers.add_parser("scan", help="Scan directory for biometric files")
    scan_parser.add_argument("path", nargs="?", help="Path to phone dump directory")
    scan_parser.add_argument("--recursive", "-r", action="store_true", 
                             help="Full recursive scan (default: top-level only)")
    scan_parser.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    scan_parser.add_argument("--custom-patterns", "-c", 
                             help="JSON file with custom detection patterns")
    scan_parser.add_argument("--parallel", "-p", type=int, default=4,
                             help="Number of parallel workers (default: 4)")
    scan_parser.add_argument("--no-extract", action="store_true",
                             help="Disable data extraction from files")
    
    return parser


def list_biometric_types(config: BiometricConfig):
    print("Supported Biometric Types:")
    print("-" * 50)
    for bt, conf in config.biometric_types.items():
        print(f"\n{bt.upper()}")
        print(f"  Confidence: {conf.get('confidence', 'medium')}")
        print(f"  Extensions: {', '.join(conf.get('extensions', []))}")
        print(f"  Patterns: {len(conf.get('patterns', []))} patterns defined")


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    config = BiometricConfig()
    
    if args.list_types:
        list_biometric_types(config)
        return 0
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "scan":
        if not args.path:
            parser.print_help()
            return 1
        
        dump_path = Path(args.path)
        if not dump_path.exists():
            print(f"Error: Path does not exist: {args.path}", file=sys.stderr)
            return 1
        
        if args.custom_patterns:
            config.load_custom_patterns(args.custom_patterns)
        
        print(f"Scanning: {args.path}")
        print(f"Recursive: {args.recursive}")
        print(f"Parallel workers: {args.parallel}")
        print("-" * 50)
        
        scanner = BiometricScanner(
            config=config,
            parallel_workers=args.parallel,
            extract_data=not args.no_extract
        )
        
        scan_results = scanner.scan(str(dump_path), recursive=args.recursive)
        
        formatter = JsonFormatter(pretty=True)
        
        results = [
            BiometricMatch(
                file=r["file"],
                relative_path=r["relative_path"],
                biometric_type=r["biometric_type"],
                confidence=r["confidence"],
                detection_methods=r["detection_methods"],
                size_bytes=r["size_bytes"],
                extracted_data=r.get("extracted_data", {})
            )
            for r in scan_results["results"]
        ]
        
        scan_info = ScanInfo(
            path=scan_results["scan_info"]["path"],
            recursive=scan_results["scan_info"]["recursive"],
            timestamp=scan_results["scan_info"]["timestamp"],
            files_scanned=scan_results["scan_info"]["files_scanned"],
            files_matched=scan_results["scan_info"]["files_matched"]
        )
        
        output = formatter.format(scan_info, results)
        
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(output)
            print(f"\nResults written to: {args.output}")
        else:
            print(output)
        
        print("-" * 50)
        print(f"Files scanned: {scan_info.files_scanned}")
        print(f"Biometric files found: {scan_info.files_matched}")
        
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
