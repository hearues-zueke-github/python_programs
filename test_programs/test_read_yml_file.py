#! /usr/bin/python3.9

import yaml

with open('test.yml') as f:
    yml_content = yaml.load(f, Loader=yaml.FullLoader)

print(yml_content)
