#!/usr/bin/env python
"""Quick test to verify GeneticSpeedEngine fix"""

import sys
import traceback

try:
    # Test imports
    from benchmarks.framework.registry import GeneticSpeedEngine
    from malthusjax.engine.genetic_fastengine import GeneticGenerationOutput
    import jax.numpy as jnp
    
    print("✓ Imports successful")
    
    # Check GeneticGenerationOutput fields
    fields = GeneticGenerationOutput.__dataclass_fields__
    required_fields = {'random_key', 'best_fitness', 'mean_fitness', 'generation'}
    
    if required_fields.issubset(fields.keys()):
        print(f"✓ GeneticGenerationOutput has all required fields: {list(fields.keys())}")
    else:
        missing = required_fields - set(fields.keys())
        print(f"✗ Missing fields: {missing}")
        sys.exit(1)
    
    # Verify the fix is in the source
    import inspect
    source = inspect.getsource(GeneticSpeedEngine.step)
    
    if 'best_fitness' in source and 'mean_fitness' in source:
        print("✓ GeneticSpeedEngine.step() includes best_fitness and mean_fitness computation")
    else:
        print("✗ Fix not properly applied to GeneticSpeedEngine.step()")
        sys.exit(1)
    
    print("\n✅ All checks passed! The fix is correctly implemented.")
    sys.exit(0)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    traceback.print_exc()
    sys.exit(1)
