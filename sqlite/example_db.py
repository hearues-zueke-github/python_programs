#! /usr/bin/python3.9.5

import os
import sqlite3

import numpy as np
import pandas as pd

# own
import constants

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
    

def get_df(conn, sql_query):
    cur = conn.cursor()
    cur.execute(sql_query)
    l_column = [x[0] for x in cur.description]
    l_data = cur.fetchall()
    df = pd.DataFrame(data=l_data, columns=l_column, dtype=object)
    
    return df


def get_d_df(conn):
    d_df = {}

    sql_query_template = "SELECT * FROM {table_name}"

    print("Getting 'sqlite_master' Table.")
    df_master = get_df(conn=conn, sql_query=sql_query_template.format(table_name='sqlite_master'))
    d_df['sqlite_master'] = df_master
    globals()['df_master'] = df_master

    for name, v_type in df_master[['name', 'type']].values:
        if v_type != 'table':
            continue
        
        print("Getting '{name}' Table.".format(name=name))
        df_master = get_df(conn=conn, sql_query=sql_query_template.format(table_name=name))
        d_df[name] = df_master

    return d_df


if __name__=='__main__':
    conn = sqlite3.connect(os.path.join(constants.file_path_sqlite_db))
    
    d_df = get_d_df(conn=conn)

    # conn = sqlite3.connect('company.sqlite')
    # cur = conn.cursor()
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
