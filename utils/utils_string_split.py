def split_quote_string(s):
    # print("s: '{}'".format(s))
    if s == '':
        return []

    l = []
    s_acc = ''
    length = len(s)
    i = 0
    is_quote = False

    while i < length:
        c = s[i]
        if c == '"' and not is_quote:
            is_quote = True
        elif c == '"' and is_quote:
            is_quote = False
        elif c == ',' and is_quote or c != ',':
            s_acc += c
        elif c == ',':
            l.append(s_acc)
            s_acc = ''
        i += 1
    l.append(s_acc)

    assert is_quote == False

    return l


def split_doublequote_string(s):
    if s == '':
        return []

    assert s[0] == '"' and s[-1] == '"'

    if '""' not in s[1:-1]:
        return None

    i = 1
    len_s = len(s)
    while i < len_s - 1:
        if s[i] == '"':
            i += 1
            assert s[i] == '"'

        i += 1

    s = s[1:-1].replace('""', '"')
    try:
        l_split = s.split('"')
        print("l_split_2: {}".format(l_split))
        # for i in range(1, len(l_split), 2):
        #     assert l_split[i].count(',') == 1
        l_split_dot = [l.split(',') if i % 2 == 0 else l.replace(',', '.') for i, l in enumerate(l_split, 0)]
        print("l_split_dot: {}".format(l_split_dot))
    except:
        print("s4: {}".format(s))

        globals()['global_vars'] = {'s': s}
        sys.exit()

    l_combine = []
    for i, l in enumerate(l_split_dot, 0):
        if i % 2 == 0:
            l_combine.extend(l)
        else:
            l_combine.append(l)

    print("s2: '{}', l_combine: {}".format(s, l_combine))
    return l_combine


assert split_quote_string('')==[]
assert split_quote_string('","')==[',']
assert split_quote_string('"",')==['', '']
assert split_quote_string(',')==['', '']
assert split_quote_string('"",""')==['', '']
assert split_quote_string('"",,""')==['', '', '']
assert split_quote_string('"",,')==['', '', '']
assert split_quote_string('"","",')==['', '', '']
assert split_quote_string('"2",,')==['2', '', '']
assert split_quote_string('"2,",,')==['2,', '', '']
assert split_quote_string('"2,3",,"3,4,5",')==['2,3', '', '3,4,5', '']
assert split_quote_string(',"2,3","3,4,5",')==['', '2,3', '3,4,5', '']
assert split_quote_string(',"2,3",,"3,4,5"')==['', '2,3', '', '3,4,5']


# assert split_doublequote_string('')==[]
# assert split_doublequote_string('""')==None
# assert split_doublequote_string('"""')==None
# assert split_doublequote_string('""""')==None
# assert split_doublequote_string('"3""4,2"')==['3', '4.2']
