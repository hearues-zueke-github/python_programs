#! /usr/bin/python3.6

if __name__ == "__main__":
    with open("num.bin", "wb") as fout:
        fout.write(bytes([0, 0, 0, 0]))

    import B
