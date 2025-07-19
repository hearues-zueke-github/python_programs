#! /usr/bin/python3.6

if __name__ == "__main__":
    a = 5
    print("a: {}".format(a))

    if a > 3:
        print("This line SHOULD be printed!")
    else:
        print("This line should NOT be printed!!!!")

    if (a:=2) > 3:
        print("This line should NOT be printed!!!!")
    else:
        print("This line SHOULD be printed!")
