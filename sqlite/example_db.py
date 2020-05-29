#! /usr/bin/python3.6

import sqlite3

if __name__=='__main__':
    print("Hello World!")

    conn = sqlite3.connect('company.sqlite')
    cur = conn.cursor()
    cur.execute('DROP TABLE IF EXISTS employee')
    cur.execute('CREATE TABLE employee (empid INT, ename TEXT, designation TEXT) ')
    cur.execute('INSERT INTO employee (empid, ename, designation) VALUES (?,?,?)', (210, 'Ahmed', 'Instructor' ))
    cur.execute('INSERT INTO employee (empid, ename, designation) VALUES (?,?,?)', (215, 'Shakeel', 'Assistant Professor' ))
    conn.commit()

    cur.execute('DROP TABLE IF EXISTS employee2')
    cur.execute('CREATE TABLE employee2 (empid INT, ename TEXT, designation TEXT) ')
    cur.execute('INSERT INTO employee2 (empid, ename, designation) VALUES (?,?,?)', (210, 'Ahmed', 'Instructor' ))
    cur.execute('INSERT INTO employee2 (empid, ename, designation) VALUES (?,?,?)', (215, 'Shakeel', 'Assistant Professor' ))
    conn.commit()

    cur.execute('SELECT * FROM employee')
    l = [v for v in cur]
    print("l: {}".format(l))

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    l = [v for v in cur]
    print("l: {}".format(l))

    conn.close()
