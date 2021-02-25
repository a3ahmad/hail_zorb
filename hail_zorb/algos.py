import torch
import torch.nn.functional as F

from . import modules
from . import utils

def fit(model, x, y):
    with torch.no_grad():
        # Make the initial predictions, and build the DAG
        predictions = model(x)

        # Breadth-first traversal from input to determine solve order
        BFS = utils.DAGQueue()
        solve_order = []

        # Load the first set of modules
        for p in x.destinations:
            if p[0] not in BFS.queue:
                BFS.push(p[0])

        # BFS traversal
        while not BFS.empty():
            m = BFS.pop()
            for p in m.destinations:
                # If this was already the solve order, push it back as other modules will need to run first
                if p[0] in solve_order:
                    solve_order.remove(p[0])
                if p[0].trainable():
                    solve_order.append(p[0])

        # For each module to solve
        for module_to_train in solve_order:
            # ANIS TODO: Run backwards to the module in question
                # ANIS TODO: Run from module_to_train to the prediction, and store the order recursively
                # ANIS TODO: Run the list back module_to_train

            # ANIS TODO: Update its weights

            # Remove module as it is now solved
            solve_order.remove(module_to_train)
            module_to_train.set_solved()

            # Re-running the prediction/inference serves three (oft-optional) goals,
            # meaning removing this could be an optimization.
            # 1) Caching the inputs on trainable modules that are up to be solved.
            # This isn't ultimately necessary, we could cache all of them always, at
            # the expense of memory.
            # 2) Update scaled activation functions (we don't need to use those)
            # 3) To track improvements to the loss over time (optional)
            predictions = model(x)
