#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys
import tarfile

import numpy as np

from PIL import Image

def create_from_tar_new_tar():
    # print("Hello World!")

    tarfile_path = 'images/pixabay_com_2.tar'
    ftar = tarfile.open(tarfile_path, 'r:*')

    tarfile_path_new = 'images/pixabay_com_new_02.tar'
    ftar_new = tarfile.open(tarfile_path_new, 'w')
    
    members = ftar.getmembers()
    root_path = os.path.commonprefix(ftar.getnames())+'/'


    # TODO: create tar objects for each pixabay folder!
    for i, member in enumerate(members, 0):
    # for i, member in enumerate(members[:1000], 0):
        if not 'jpg' in member.path:
            continue

        print("i: {}".format(i))
        file_name = member.name.split("/")[-1]
        info = tarfile.TarInfo(name=file_name)
        f = ftar.extractfile(member)
        bytes_read = f.read()
        f.seek(0)
        info.size = len(bytes_read)
        ftar_new.addfile(tarinfo=info, fileobj=f)

    print("tarfile_path: {}".format(tarfile_path))
    print("root_path: {}".format(root_path))
    print("len(members): {}".format(len(members)))
    
    ftar.close()
    ftar_new.close()


def create_tar_jpg_files_from_dir_path(dir_path, tar_file_path):
    if "/" != dir_path[-1]:
        dir_path += "/"
    root_dir, _, files_name = next(os.walk(dir_path))
    ftar = tarfile.open(tar_file_path, "w")
    for i, file_name in enumerate(files_name, 0):
        print("i: {}".format(i))
        # file_name = member.name.split("/")[-1]
        if not '.jpg' in file_name:
            print("Skip file '{}'!".format(file_name))
            continue

        with open(root_dir+file_name, "rb") as f:
            info = tarfile.TarInfo(name=file_name)
            bytes_read = f.read()
            f.seek(0)
            info.size = len(bytes_read)
            ftar.addfile(tarinfo=info, fileobj=f)

    ftar.close()


def convert_image_to_smaller_pix(img, w_resize, h_resize):
    pix = np.array(img)

    if len(pix.shape) < 2:
        raise Exception

    h, w = pix.shape[:2]
    if len(pix.shape) == 2:
        pix = np.tile(pix.reshape((-1, )), 3).reshape((3, -1)).T.reshape((h, w, 3))
    elif pix.shape[2] == 4:
        pix = pix[:, :, :3]

    if h*4 <= w*3: # trim horizontally
        hn = h
        wn = int(h*4/3)
        x = int((w-wn)/2)
        y = 0
    else:          # trim vertically
        wn = w
        hn = int(w*3/4)
        y = int((h-hn)/2)
        x = 0

    pixn = pix[y:y+hn, x:x+wn]
    imgn = Image.fromarray(pixn)

    # imgnr = imgn.resize((60, 45), Image.LANCZOS)
    imgnr = imgn.resize((w_resize, h_resize), Image.LANCZOS)
    # imgnr = imgn.resize((200, 150), Image.LANCZOS)
    pixnr = np.array(imgnr)

    return pixnr


def combine_from_tar_to_tar_new_files(file_template_in, file_template_out):
    print("file_template_in: {}".format(file_template_in))
    print("file_template_out: {}".format(file_template_out))

    if not '.tar' in file_template_in:
        print("Need '.tar' extension in file_template_in!")
        sys.exit(-3)
    if not '.tar' in file_template_out:
        print("Need '.tar' extension in file_template_out!")
        sys.exit(-3)

    file_template_in = "images/"+file_template_in
    file_template_out = "images/"+file_template_out

    was_found_zeros_in = False
    for i_in in range(1, 11):
        if "_"+"0"*i_in+"." in file_template_in:
            was_found_zeros_in = True
            break
    if not was_found_zeros_in:
        print("Need '_0..0.' in file_template_in for replace format!")
        sys.exit(-4)
    file_template_in = file_template_in.replace("_"+"0"*i_in+".", "_{{:0{num}}}.".format(num=i_in))
    print("file_template_in: {}".format(file_template_in))

    was_found_zeros_out = False
    for i_out in range(1, 11):
        if "_"+"0"*i_out+"." in file_template_out:
            was_found_zeros_out = True
            break
    if not was_found_zeros_out:
        print("Need '_0..0.' in file_template_out for replace format!")
        sys.exit(-4)
    file_template_out = file_template_out.replace("_"+"0"*i_out+".", "_{{:0{num}}}.".format(num=i_out))
    print("file_template_out: {}".format(file_template_out))

    tar_files_path = []
    for i in range(1, 100):
        tar_file_path = file_template_in.format(i)
        if not os.path.exists(tar_file_path):
            break
        tar_files_path.append(tar_file_path)

    tar_files_path = np.array(tar_files_path)[:3]
    print("tar_files_path: {}".format(tar_files_path))
        
    tar_file_path = file_template_in.format(1)
    
    all_files_name_lst = []
    all_members = []
    ftar_files = []
    for tar_file_path in tar_files_path:
        print("doing now: tar_file_path: {}".format(tar_file_path))
        ftar = tarfile.open(tar_file_path, 'r:*')
        ftar_files.append(ftar)
        members = ftar.getmembers()
        print("len(members): {}".format(len(members)))
        all_files_name = []
        members_part = members[:1000]
        all_members.extend(members_part)
        for j, m in enumerate(members_part, 0):
            if j%100 == 0:
                print("j: {}".format(j))
            all_files_name.append(m.name)
        all_files_name_lst.append(np.array(all_files_name))
        # all_files_name_lst.append(np.array([m.name for m in members]))

    complete_all_files_name = np.hstack(all_files_name_lst)
    print("complete_all_files_name.shape: {}".format(complete_all_files_name.shape))
    all_members = np.array(all_members)

    lst_lens = [len(l) for l in all_files_name_lst]

    dirs_idx = np.hstack([np.zeros((length, ), dtype=np.int32)+i for i, length in enumerate(lst_lens, 0)])
    print("dirs_idx.shape: {}".format(dirs_idx.shape))

    print("sort complete_all_files_name")

    idxs_argsort = np.argsort(complete_all_files_name)

    complete_all_files_name = complete_all_files_name[idxs_argsort]
    all_members = all_members[idxs_argsort]
    dirs_idx = dirs_idx[idxs_argsort]

    idxs_non_duplicates = np.hstack(((True, ), complete_all_files_name[:-1]!=complete_all_files_name[1:]))

    complete_all_files_name = complete_all_files_name[idxs_non_duplicates]
    all_members = all_members[idxs_non_duplicates]
    dirs_idx = dirs_idx[idxs_non_duplicates]

    print("complete_all_files_name: {}".format(complete_all_files_name))
    print("dirs_idx: {}".format(dirs_idx))

    w_resize = 60
    h_resize = 45
    pixses_rgb_60_45 = np.empty((complete_all_files_name.shape[0], h_resize, w_resize, 3), dtype=np.uint8)
    print("pixses_rgb_60_45.shape: {}".format(pixses_rgb_60_45.shape))

    ignore_idxs = []
    for idx, (file_name, member, dir_idx) in enumerate(zip(complete_all_files_name, all_members, dirs_idx), 0):
        if idx % 100 == 0:
            print("idx: {}".format(idx))
        ftar = ftar_files[dir_idx]
        f = ftar.extractfile(member)
        # bytes_read = f.read()
        img = Image.open(f)
        # img = Image.open(bytes_read)
        
        try:
            pix = convert_image_to_smaller_pix(img, 60, 45)
            # pix = np.array(img)
            # f.seek(0)
            pixses_rgb_60_45[idx] = pix
        except:
            ignore_idxs.append(idx)
        # print("tar_files_path[dir_idx]: {}".format(tar_files_path[dir_idx]))
        # print("file_name: {}".format(file_name))
        # print("member: {}".format(member))
        # print("dir_idx: {}".format(dir_idx))
        # print("ftar: {}".format(ftar))
        # # print("len(bytes_read): {}".format(len(bytes_read)))
        # print("pix.shape: {}".format(pix.shape))
        # img.show()
        # break

    # TODO: write maybe multiprocessing file extractor!
    with gzip.open('datas/other_pixses_60_45.pkl.gz', 'wb') as f:
        dill.dump(pixses_rgb_60_45, f)

    for ftar in ftar_files:
        ftar.close()

    # example call:
    # creating_reading_tar_files.py combine_new_tar_files pixabay_com_new_00.tar pixabay_com_new_combine_00.tar


if __name__ == "__main__":
    argv = sys.argv

    valid_options = ['create_tar_one', 'combine_new_tar_files']

    if len(argv) < 2:
        print("Need one of followging options:")
        for option in valid_options:
            print(" - {}".format(option))
        print("Program terminating...")

        print("exit code -1")
        sys.exit(-1)

    option = argv[1]

    if option == "create_tar_one":
        if len(argv) < 4:
            print("Wrong call of program!")
            print("Need. ./<program_name> create_tar_one <dir_path> <tar_file_path>")
            sys.exit(-1)

        dir_path = argv[2]
        tar_file_path = argv[3]

        create_tar_jpg_files_from_dir_path(dir_path, tar_file_path)
    elif option == "combine_new_tar_files":
        if len(argv) < 4:
            print("Wrong call of program!")
            print("Need. ./<program_name> combine_new_tar_files <file_template_in> <file_template_out>")
            sys.exit(-1)

        file_template_in = argv[2]
        file_template_out = argv[3]

        combine_from_tar_to_tar_new_files(file_template_in, file_template_out)
    else:
        print("Not a valid option! Following options are available:")
        for option in valid_options:
            print(" - {}".format(option))
        print("Program terminating...")

        print("exit code -2")
        sys.exit(-2)
