def replace_in_file(filepath, old, new):
    with open(filepath, "r") as f:
        content = f.read()
    content = content.replace(old, new)
    with open(filepath, "w") as f:
        f.write(content)


replace_in_file(
    "lsp_project/src/lsp/evaluator/interpreter.py",
    "from malthusjax.core.fitness.linear_gp_evaluator import TENSORGP_FUNCTIONS",
    "from lsp.evaluator.base import TENSORGP_FUNCTIONS",
)

replace_in_file(
    "lsp_project/src/lsp/evaluator/base.py",
    "from malthusjax.core.fitness.linear_gp_interpreter import predict_one",
    "from lsp.evaluator.interpreter import predict_one",
)

replace_in_file(
    "lsp_project/src/lsp/evaluator/prefix.py",
    "from malthusjax.core.fitness.linear_gp_evaluator import",
    "from lsp.evaluator.base import",
)

replace_in_file(
    "lsp_project/src/lsp/evaluator/prefix.py",
    "from malthusjax.core.genome.prefix.genome import",
    "from lsp.genome import",
)

print("Imports fixed.")
