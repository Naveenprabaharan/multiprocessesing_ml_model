import multiprocessing
import time
import os




def make_square(nnmbers,q):
    for i in nnmbers:
        q.put(i*i)

def make_negative(nnmbers,q):
    for i in nnmbers:
        q.put(-1*i)

if __name__=='__main__':
    q = multiprocessing.Queue()
    numbers = range(1,6)
    p1 = multiprocessing.Process(target=make_square, args=(numbers,q))
    p2 = multiprocessing.Process(target=make_negative, args=(numbers,q))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    while not q.empty():
        print(q.get())
    print('work')









# Queue multiprocessing

'''def make_square(nnmbers,q):
    for i in nnmbers:
        q.put(i*i)

def make_negative(nnmbers,q):
    for i in nnmbers:
        q.put(-1*i)

if __name__=='__main__':
    q = multiprocessing.Queue()
    numbers = range(1,6)
    p1 = multiprocessing.Process(target=make_square, args=(numbers,q))
    p2 = multiprocessing.Process(target=make_negative, args=(numbers,q))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    while not q.empty():
        print(q.get())
    print('work')
'''















# shared Array
'''def add_100(numbers,lock):
    for i in range(100):
        for _ in range(len(numbers)):
            with lock:
                numbers[_] +=1

if __name__=='__main__':
    shared_array = multiprocessing.Array('d',[0.0, 100.0, 200.0])
    lock = multiprocessing.Lock()
    print(f'oru number is start @ {shared_array[:]}')

    p1 = multiprocessing.Process(target=add_100, args=(shared_array,lock))
    p2 = multiprocessing.Process(target=add_100, args=(shared_array,lock))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    print(f'oru number is End @ {shared_array[:]}')

    print('work')'''


# shared value
'''def add_100(number,lock):
    for i in range(100):
        time.sleep(.01)
        with lock:
            number.value +=1

if __name__=='__main__':
    shared_value = multiprocessing.Value('i',0)
    lock = multiprocessing.Lock()
    print(f'oru number is start @ {shared_value.value}')
    p1 = multiprocessing.Process(target=add_100, args=(shared_value,lock))
    p2 = multiprocessing.Process(target=add_100, args=(shared_value,lock))

    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print(f'oru number is end @ {shared_value.value}')
    print('work')'''