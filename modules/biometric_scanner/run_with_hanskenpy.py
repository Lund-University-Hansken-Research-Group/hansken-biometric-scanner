#!/usr/bin/env python3
"""
Run the BiometricModelsPlugin against an extracted Hansken project using Hansken.py.
Then search for results.

Usage:
    python3 run_with_hanskenpy.py @/home/hansken/argfile
    python3 run_with_hanskenpy.py --search @/home/hansken/argfile
"""

from hansken_extraction_plugin.runtime.extraction_plugin_runner import run_with_hanskenpy
from plugin import BiometricModelsPlugin
from hansken.tool import run


def search_results(context):
    """Search for traces enriched by the plugin."""
    with context:
        print("=" * 60)
        print("Searching for misc.biometricModelType=*")
        print("=" * 60)

        results = context.search('misc.biometricModelType=*')
        count = 0
        for trace in results:
            count += 1
            print(f"\n--- Trace {count} ---")
            print(f"  uid:  {trace.uid}")
            print(f"  name: {trace.name}")
            print(f"  path: {trace.get('file.path')}")
            print(f"  misc.biometricModelType:       {trace.get('misc.biometricModelType')}")
            print(f"  misc.biometricModelFramework:   {trace.get('misc.biometricModelFramework')}")
            print(f"  misc.biometricModelDetectedBy: {trace.get('misc.biometricModelDetectedBy')}")
            print(f"  misc.biometricModelConfidence: {trace.get('misc.biometricModelConfidence')}")

        if count == 0:
            print("\nNo traces found with misc.biometricModelType set.")
            print("\nSearching for all .pkl files instead:")

            pkl_results = context.search('file.extension=pkl')
            for trace in pkl_results:
                print(f"  {trace.uid}  {trace.name}  ext={trace.get('file.extension')}")
                print(f"    misc.biometricModelType = {trace.get('misc.biometricModelType')}")

        print(f"\nTotal: {count} traces with biometric model properties")


if __name__ == '__main__':
    import sys

    defaults = dict(
        endpoint='http://127.0.0.1:9091/gatekeeper/',
        keystore='http://127.0.0.1:9090/keystore/',
        project='9e9ef48f-bb65-4800-8637-5c0b1e00550c',
    )

    if '--search' in sys.argv:
        sys.argv.remove('--search')
        run(with_context=search_results, **defaults)
    else:
        print("Step 1: Running plugin with Hansken.py...")
        print("=" * 60)
        run_with_hanskenpy(BiometricModelsPlugin)

        print("\nStep 2: Searching for results...")
        print("=" * 60)
        run(with_context=search_results, **defaults)
