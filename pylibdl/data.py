import multiprocessing as mp
import threading
import numpy as np
import queue
from pylibdl.tensor import tensor


class Dataset:

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class DataLoader:

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0,
                 drop_last: bool = False):
        if num_workers == 0:
            try:
                num_workers = mp.cpu_count()
            except NotImplementedError:
                num_workers = 1
        self.dataset = dataset
        self.num_workers = num_workers

        self.batch_idxs = []
        perm = np.random.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
        for i in range(0, len(dataset) - batch_size if drop_last else len(dataset), batch_size):
            self.batch_idxs.append(perm[i:i + batch_size])

    def __iter__(self):
        return _DataLoaderIter(self.dataset, self.num_workers, self.batch_idxs)


class DataLoaderWorker(threading.Thread):
    def __init__(self, dataset: Dataset, idx, index_queue: queue.Queue):
        super().__init__()
        self.dataset = dataset
        self.idx = idx
        self.batch_queue = index_queue

    def _prepare_batch(self, idx):
        elements = [self.dataset[i] for i in idx]
        if type(elements[0]) is not tuple:
            for i in range(len(elements)):
                elements[i] = elements[i],
        return tuple(tensor(np.stack(x, axis=-1)) for x in zip(*elements))

    def run(self):
        batch = self._prepare_batch(self.idx)
        self.batch_queue.put(batch)


class _DataLoaderIter:

    def __init__(self, dataset: Dataset, num_workers: int, batch_idxs: list):
        self.dataset = dataset
        self.index_queue = queue.Queue()
        for m in batch_idxs:
            self.index_queue.put(m)
        self.remaining_batches = self.index_queue.qsize()

        self.worker = []

        self.batch_queue = mp.Queue()
        for _ in range(num_workers):
            self._load_batch()

    def _load_batch(self):
        if self.index_queue.empty():
            return
        self.worker.append(DataLoaderWorker(self.dataset, self.index_queue.get(), self.batch_queue))
        self.worker[-1].start()

    def __next__(self):
        if self.remaining_batches <= 0:
            raise StopIteration
        self.remaining_batches -= 1
        self._load_batch()
        return self.batch_queue.get()

    def __iter__(self):
        return self

    def __del__(self):
        for m in self.worker:
            m.join()
        try:
            while True:
                self.batch_queue.get(timeout=.25)
        except queue.Empty:
            pass
        self.batch_queue.close()
        self.batch_queue.join_thread()

