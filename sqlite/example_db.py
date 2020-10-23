#! /usr/bin/python3.6

import sqlite3

import numpy as np

def do_table_name_exists(conn, table_name):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return table_name in set([v[0] for v in cur])


def create_table_colors(conn):
    cur = conn.cursor()

    cur.execute('DROP TABLE IF EXISTS colors')
    cur.execute('CREATE TABLE colors (idx INT, hex1 TEXT)')


    l_hex_str = list('0123456789ABCDEF')
    for i in range(0, 100):
        hex_str = ''.join([l_hex_str[j] for j in np.random.randint(0, len(l_hex_str), (6, ))])
        print("i: {}, hex_str: {}".format(i, hex_str))
        cur.execute('INSERT INTO colors (idx, hex1) VALUES (?,?)', (i, hex_str))

    conn.commit()



def get_whole_table(conn, table_name):
    cur = conn.cursor()
    cur.execute('SELECT * FROM {}'.format(table_name))
    l = [v for v in cur]
    return l
    

if __name__=='__main__':
    print("Hello World!")

    conn = sqlite3.connect('company.sqlite')
    cur = conn.cursor()
    # cur.execute('DROP TABLE IF EXISTS employee')
    # cur.execute('CREATE TABLE employee (empid INT, ename TEXT, designation TEXT) ')
    # cur.execute('INSERT INTO employee (empid, ename, designation) VALUES (?,?,?)', (210, 'Ahmed', 'Instructor' ))
    # cur.execute('INSERT INTO employee (empid, ename, designation) VALUES (?,?,?)', (215, 'Shakeel', 'Assistant Professor' ))
    # conn.commit()

    # cur.execute('DROP TABLE IF EXISTS employee2')
    # cur.execute('CREATE TABLE employee2 (empid INT, ename TEXT, designation TEXT) ')
    # cur.execute('INSERT INTO employee2 (empid, ename, designation) VALUES (?,?,?)', (210, 'Ahmed', 'Instructor' ))
    # cur.execute('INSERT INTO employee2 (empid, ename, designation) VALUES (?,?,?)', (215, 'Shakeel', 'Assistant Professor' ))
    # conn.commit()

    # cur.execute('SELECT * FROM employee')
    # l = [v for v in cur]
    # print("l: {}".format(l))

    # cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    # l = [v for v in cur]
    # print("l: {}".format(l))

    # conn.close()
