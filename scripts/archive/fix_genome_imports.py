import os

def process_genome_base():
    filepath = "lsp_project/src/lsp/genome/base.py"
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Strip everything from ConstantGenomeConfig down
    idx = content.find("@struct.dataclass\nclass ConstantGenomeConfig")
    if idx != -1:
        content = content[:idx]
    
    # Imports
    content = content.replace(
        "from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig",
        "from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig"
    )
    
    with open(filepath, 'w') as f:
        f.write(content)

def process_population():
    filepath = "lsp_project/src/lsp/genome/population.py"
    with open(filepath, 'r') as f:
        content = f.read()
        
    content = content.replace(
        "from malthusjax.core.genome.prefix.genome import BasePrefixAwareGenome, PrefixGenomeConfig",
        "from lsp.genome.base import BasePrefixAwareGenome, PrefixGenomeConfig"
    )
    with open(filepath, 'w') as f:
        f.write(content)

process_genome_base()
process_population()
print("Genome imports fixed.")
