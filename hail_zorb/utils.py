
class DAGQueue:
    def __init__(self):
        self.queue = []

    def push(self, x):
        self.queue.append(x)

    def pop(self):
        for x in self.queue:
            # If this has no parents, it can be removed
            if x.sources is None:
                break

            # If all parents don't exist in the queue (they've been processed), it
            # can be removed
            parents_processed = sum([0 if p[0] in self.queue else 1 for p in x.sources])
            if parents_processed == len(x.sources):
                break

        # Remove the item from the queue, meaning that it is processed
        self.queue.remove(x)
        return x

    def empty(self):
        return self.queue.empty()


class DAGReverseQueue:
    def __init__(self):
        self.queue = []

    def push(self, x):
        self.queue.append(x)

    def pop(self):
        for x in self.queue:
            # If this has no parents, it can be removed
            if len(x.destinations) == 0:
                break

            # If all parents don't exist in the queue (they've been processed), it
            # can be removed
            parents_processed = sum([0 if p[0] in self.queue else 1 for p in x.destinations.values()])
            if parents_processed == len(x.destinations):
                break

        # Remove the item from the queue, meaning that it is processed
        self.queue.remove(x)
        return x

    def empty(self):
        return self.queue.empty()
