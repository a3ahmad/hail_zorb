import torch
import torch.nn.functional as F

from . import modules
from . import utils


def traverse_to_output(module, module_collection=set()):
    module_collection.add(module)

    for destination in module.destinations:
        traverse_to_output(destination, module_collection)

    return module_collection


def backward_to_module(module, outputs, module_collection):
    BFS = utils.DAGReverseQueue()
    solve_order = []

    # Load the first set of modules
    for output in outputs:
        if output.source is None:
            continue
        if (output.source[0] in module_collection) and (output.source[0] not in BFS.queue):
            BFS.push(output.source[0])

    # BFS traversal, constrainted to modules in module_collection
    while not BFS.empty():
        m = BFS.pop()

        if (m.source is not None) and (m.source[0] in module_collection):
            # If this was already the solve order, push it back as other modules will
            # need to run first
            if m.source[0] in solve_order:
                solve_order.remove(m.source[0])
            if m.source[0].trainable():
                solve_order.append(m.source[0])

    for module_to_solve in solve_order:
        if module_to_solve == module:
            break

        module_to_solve.solve()


def fit(model, x, y):
    with torch.no_grad():
        # Make the initial predictions, and build the DAG
        predictions = model(x)

        # Breadth-first traversal from input to determine solve order
        BFS = utils.DAGQueue()
        train_order = []

        # Load the first set of modules
        for input in x:
            for p in input.destinations:
                if p[0] not in BFS.queue:
                    BFS.push(p[0])

        # BFS traversal
        while not BFS.empty():
            m = BFS.pop()
            for p in m.destinations:
                # If this was already the solve order, push it back as other modules will
                # need to run first
                if p[0] in train_order:
                    train_order.remove(p[0])
                if p[0].trainable():
                    train_order.append(p[0])

        # For each module to solve
        for module_to_train in train_order:
            # Run from module_to_train to the predictions, without executing and store
            # the traversed Modules
            traversed_modules = traverse_to_output(module_to_train)

            # Run from predictions backwards, only visiting the Modules traversed above
            backward_to_module(module_to_train, predictions, traversed_modules)

            # Update its weights
            module_to_train.update()

            # Remove module as it is now solved
            train_order.remove(module_to_train)
            module_to_train.set_solved()

            # Re-running the prediction/inference serves three (oft-optional) goals,
            # meaning removing this could be an optimization.
            # 1) Caching the inputs on trainable modules that are up to be solved.
            # This isn't ultimately necessary, we could cache all of them always, at
            # the expense of memory.
            # 2) Update scaled activation functions (we don't need to use those)
            # 3) To track improvements to the loss over time (optional)
            predictions = model(x)
