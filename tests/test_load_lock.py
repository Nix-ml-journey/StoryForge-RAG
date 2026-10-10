import threading
import time

from storyforge._load_lock import serialized


def test_two_threads_load_a_cached_model_only_once():
    cache, loads = {}, []

    @serialized
    def get():
        if "m" not in cache:
            loads.append(1)
            time.sleep(0.05)  # a slow model load; without the lock the second thread would also load
            cache["m"] = object()
        return cache["m"]

    threads = [threading.Thread(target=get) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(loads) == 1
