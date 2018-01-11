#! /usr/bin/python3

import gzip
import os
import subprocess
import time

import numpy as np
import pickle as pkl

def clear():
    if os.name in ('nt','dos'):
        subprocess.call("cls")
    elif os.name in ('linux','osx','posix'):
        subprocess.call("clear")
    else:
        print("\n"*120)

class temp:
    def __init__(self):
        pass

if __name__ == "__main__":
    num_pi = "1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679" + \
             "8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196" + \
             "4428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273" + \
             "7245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094" + \
             "3305727036575959195309218611738193261179310511854807446237996274956735188575272489122793818301194912" + \
             "9833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132" + \
             "0005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235" + \
             "4201995611212902196086403441815981362977477130996051870721134999999837297804995105973173281609631859" + \
             "5024459455346908302642522308253344685035261931188171010003137838752886587533208381420617177669147303" + \
             "5982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989"
    num_pi_groups = list(map(lambda x: num_pi[5*x[0]: 5*x[1]], zip(range(0, 199), range(1, 200))))
    print("len(num_pi): {}".format(len(num_pi)))
    for i, group in enumerate(num_pi_groups):
        print("i: {}, group: {}".format(i, group))

    file_name = "number_pi_statistics.pkl.gz"
    if not os.path.exists(file_name):
        stats = temp()
        stats.false_groups = np.zeros((100, ))
        stats.attempt = {"correct": [], "false": [], "time": []}
        with gzip.open(file_name, "wb") as fout:
            pkl.dump(stats, fout)

    correct = 0
    false = 0

    false_groups = np.zeros((100, ))
    # TODO: choose a challange for 5, 10, 20, 50, 100 groups
    #       every challenge can have parts of groups e.g. for 5:
    #       groups
    #       (1) 1-5
    #       (2) 6-10
    #       (3) 11-15 etc.
    
    clear()
    group_sizes = [5, 10, 20, 50, 100]
    print("What group size do you want?")
    for i, s in enumerate(group_sizes):
        print("({}) {}".format(i+1, s))
    choosen_group_size = group_sizes[int(input("Choose between 1-{}: ".format(len(group_sizes))))-1]

    print("What groups do you want to do?")
    for i in range(100//choosen_group_size):
        print("({}) {}-{}".format(i+1, choosen_group_size*i+1, choosen_group_size*(i+1)))
    choosen_groups = int(input("Choose between 1-{}: ".format(100//choosen_group_size)))
    
    first_group = (choosen_groups-1)*choosen_group_size
    last_group = choosen_groups*choosen_group_size

    clear()
    start_time = time.time()
    print("Give pi groups between {}-{}".format(first_group+1, last_group))
    # for i in range(0, 20):
    idx_prev = np.random.randint(first_group, last_group)
    idx_prev_prev = np.random.randint(first_group, last_group)
    is_last_false = False
    # for i in range(0, 50):
    i = 0
    while i < 50:
        print("Round {}.".format(i+1), end="")

        if not is_last_false:
            while True:
                idx = np.random.randint(first_group, last_group)
                if idx != idx_prev and idx != idx_prev_prev:
                    idx_prev_prev = idx_prev
                    idx_prev = idx
                    break
            group = num_pi_groups[idx]
        else:
            print(" Try again!!", end="")

        user_input = input("\nWhat is the group idx \x1b[1;32;34m{:3}\x1b[0m? \x1b[0;33;33m".format(idx+1))
        print("\x1b[0m")

        clear()
        while len(user_input) != 5:
            print("Wrong Input!! Try again! Input was {}5 characters".format(">" if len(user_input) > 5 else "<"))
            print("Round {}. Try again!!".format(i+1))
            user_input = input("What is the group idx \x1b[1;32;30m{:3}\x1b[0m? \x1b[0;33;33m".format(idx+1))
            print("\x1b[0m")
            clear()

        if group == user_input:
            if is_last_false:
                is_last_false = False            
            else:
                correct += 1
            print("You answer is correct!")
        else:
            if not is_last_false:
                false_groups[idx] += 1
                false += 1
            is_last_false = True
            i -= 1
            print("You answer is false! your answer: {}, correct answer is: {} for idx {}".format(user_input, group, idx+1))

        i += 1

    end_time = time.time()
    taken_time = end_time-start_time
    print("Correct answers: {}".format(correct))
    print("False answers: {}".format(false))
    print("Taken time: {:0.5f} sec".format(taken_time))

    with gzip.open(file_name, "rb") as fin:
        stats = pkl.load(fin)

    stats.false_groups += false_groups
    stats.attempt["correct"].append(correct)
    stats.attempt["false"].append(false)
    stats.attempt["time"].append(taken_time)

    print("all false per group:\n{}".format(stats.false_groups))
    print("all attempts:\n{}".format(stats.attempt))

    with gzip.open(file_name, "wb") as fout:
        pkl.dump(stats, fout)
