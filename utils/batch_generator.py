# -*- coding: utf-8 -*-
#Loading images with CPU background threads during GPU forward passes saves a lot of time
#Credit: J. Schlüter (https://github.com/Lasagne/Lasagne/issues/12)
def threadedBatchGenerator(generator, num_cached=10):
    
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    #define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    #start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    #run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()
