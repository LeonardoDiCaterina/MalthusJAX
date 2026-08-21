import glob

def fix_imports(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Change back to from lsp.genome import ... instead of lsp.genome.base
    content = content.replace("from lsp.genome.base import", "from lsp.genome import")
    
    with open(filepath, 'w') as f:
        f.write(content)

for filepath in glob.glob("lsp_project/tests/*.py"):
    fix_imports(filepath)

print("Test imports fixed again.")
