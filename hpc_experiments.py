import numpy as np
import threading
import queue
import time
import matplotlib.pyplot as plt
import logging

def hpc_thread_function_w(a, w, h, cross_queues, out_queue, i, n_row, n_col):
    counter = 0
    my_u = h @ h.T # calculate my piece of u
    u = h @ h.T # keep a running sum (not a reference to my_u!!)
    status = [0 for _ in cross_queues] # 0 means nothing done yet
    while sum(status) < 2*len(cross_queues): # 2 means done
        for index, (first, second, q, lock) in enumerate(cross_queues):
            with lock:
                if status[index] == 0 and q.empty():
                    q.put(my_u)
                    status[index] = 1 # 1 means in write-first mode
                elif status[index] == 0 and not q.empty():
                    other_u = q.get()
                    assert q.empty()
                    u += other_u
                    q.put(my_u)
                    q.put(None) # mark that content has been read and re-written by filling queue to 2
                    assert q.full()
                    status[index] = 2 # 2 means done
                elif status[index] == 1 and q.full(): # see that content has been read and re-written
                    other_u = q.get()
                    q.get() # clear the cross-queue
                    assert q.empty()
                    u += other_u
                    status[index] = 2 # 2 means done

    h_block_pieces = [(i, h)]
    status = [0 for _ in cross_queues] # 0 means nothing done yet
    while sum(status) < 2*(((len(cross_queues)+1)/n_col)-1): # 2 means done. There are len(cross_queues) threads, one out of each column needs to be 2, minus myself.
        for index, (first, second, q, lock) in enumerate(cross_queues):
            if int(first / n_row) == int(second / n_row): # check if in the same block # POTENTIAL BUGG HERE??
                with lock:
                    if status[index] == 0 and q.empty():
                        q.put((i, h))
                        status[index] = 1 # 1 means in write-first mode
                    elif status[index] == 0 and not q.empty() and not q.full(): # add extra check to avoid edge case race condition
                        h_block_pieces.append(q.get())
                        q.put((i, h))
                        q.put(None) # mark that content has been read and re-written by filling queue to 2
                        status[index] = 2 # 2 means done
                    elif status[index] == 1 and q.full(): # see that content has been read and re-written
                        h_block_pieces.append(q.get())
                        q.get() # clear the cross-queue
                        status[index] = 2 # 2 means done
            #logging.warning("silly bug.")
    h_block = np.hstack([x[1] for x in sorted(h_block_pieces, key = lambda x: x[0])])

    my_v = a @ h_block.T
    v_block_pieces = [(i, a @ h_block.T)]
    status = [0 for _ in cross_queues] # 0 means nothing done yet
    while sum(status) < 2*len(cross_queues):
        for index, (first, second, q, lock) in enumerate(cross_queues):
            if int(first / n_row) == int(second / n_row): # check if in the same block # POTENTIAL BUGG HERE???
                with lock:
                    if status[index] == 0 and q.empty():
                        q.put((i, my_v))
                        status[index] = 1 # 1 means in write-first mode
                    elif status[index] == 0 and not q.empty() and not q.full(): # add extra check to avoid edge case race condition
                        v_block_pieces.append(q.get())
                        q.put((i, my_v))
                        q.put(10000) # mark that content has been read and re-written by filling queue to 2
                        status[index] = 2 # 2 means done
                    elif status[index] == 1 and q.full(): # see that content has been read and re-written
                        v_block_pieces.append(q.get())
                        q.get() # clear the cross-queue
                        status[index] = 2 # 2 means done
            elif len(v_block_pieces) == n_row: # if not in the same block, wait until all entries are established, then start adding
                with lock:
                    if status[index] == 0 and q.empty():
                        q.put((i, my_v))
                        status[index] = 1 # 1 means in write-first mode
                    elif status[index] == 0 and not q.empty() and not q.full(): # add extra check to avoid edge case race condition
                        other_i, other_v = q.get()
                        v_block_pieces[int(other_i / n_row)] = (v_block_pieces[int(other_i / n_row)][0], v_block_pieces[int(other_i / n_row)][1] + other_v)
                        q.put((i, my_v))
                        q.put(None) # mark that content has been read and re-written by filling queue to 2
                        status[index] = 2 # 2 means done
                    elif status[index] == 1 and q.full(): # see that content has been read and re-written
                        other_i, other_v = q.get()
                        v_block_pieces[int(other_i / n_row)] = (v_block_pieces[int(other_i / n_row)][0], v_block_pieces[int(other_i / n_row)][1] + other_v)
                        q.get() # clear the cross-queue
                        status[index] = 2 # 2 means done
    v_block = np.vstack([x[1] for x in sorted(v_block_pieces, key = lambda x: x[0])])


    
    
    # # out_queue.put(u)
    # # out_queue.put((int(i / n_col), h_block))
    out_queue.put(v_block)

def HPC_NMF(a, k, p_row, p_col, numIter):
    m, n = np.shape(a)
    if m % (p_row*p_col) > 0:
        raise TypeError('Input first dimension not divisible by number of threads')
    if n % (p_row*p_col) > 0:
        raise TypeError('Input second dimension not divisible by number of threads')
    w = np.random.rand(m, k)
    h = np.random.rand(k, n)

    a_pieces = [np.split(x, p_col, 1) for x in np.split(a, p_row, 0)] # cut a into p_row x p_col pieces of shape m/p_row x n/p_col
    assert np.shape(a_pieces[0][0]) == (int(m/p_row), int(n/p_col))

    for _ in range(numIter):
        u = np.zeros((k, k))
        h_pieces = np.split(h, p_row*p_col, 1) # cut h into p_row*p_col pieces of shape k x n/(p_row*p_col)
        assert np.shape(h_pieces[0]) == (k, int(n/(p_row*p_col)))

        threads_w = []
        cross_queues_w = []
        out_queue_w = queue.Queue(maxsize=p_row*p_col)
        for first in range(p_row*p_col):
            for second in range(first+1, p_row*p_col):
                cross_queues_w.append((first, second, queue.Queue(maxsize=2), threading.Lock())) # assign each pair a unique queue and a unique lock
        for i in range(p_row*p_col): # split into p threads to calculate updates for each piece
            newThread = threading.Thread(target = hpc_thread_function_w, args = 
                (a_pieces[i%p_row][int(i/p_row)],
                 w,
                 h_pieces[i],
                 [(first, second, q, l) for first, second, q, l in cross_queues_w if i == first or i == second],
                 out_queue_w,
                 i,
                 p_row,
                 p_col))
            newThread.start()
            threads_w.append(newThread)
        for thread in threads_w: # wait for all threads to complete
            thread.join()
        

        # while not out_queue_w.empty():
        #     assert np.all(np.isclose(out_queue_w.get(), h @ h.T))

        # h_pieces = []
        # covered = set()
        # while not out_queue_w.empty():
        #     index, piece = out_queue_w.get()
        #     if not index in covered:
        #         h_pieces.append((index, piece))
        #         covered.add(index)
        # assert np.all(np.isclose(h, np.hstack([x[1] for x in sorted(h_pieces, key = lambda x : x[0])])))

        while not out_queue_w.empty():
            print(out_queue_w.get())
        print('')
        print(a@h.T)
    return

HPC_NMF(np.identity(4)+1, 4, 2, 2, 1)
