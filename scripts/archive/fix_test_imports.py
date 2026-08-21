import os
import glob

def fix_imports(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    content = content.replace("from lsp.genome import", "from lsp.genome.base import")
    content = content.replace("from lsp.crossover import", "from lsp.operators.crossover import")
    content = content.replace("from lsp.mutation import", "from lsp.operators.mutation import")
    content = content.replace("from lsp.selection import", "from lsp.operators.selection import")
    # also rename LinearScheduledMutation to AnnealedTopologicalMutation in tests
    content = content.replace("LinearScheduledMutation", "AnnealedTopologicalMutation")
    
    with open(filepath, 'w') as f:
        f.write(content)

for filepath in glob.glob("lsp_project/tests/*.py"):
    fix_imports(filepath)

print("Test imports fixed.")
