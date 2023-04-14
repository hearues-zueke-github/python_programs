import yaml
import os

fn main() {
	filename := './sample_data.yaml'

    data := os.read_file(filename) or {
        panic('error reading file $filename')
        return
    }
    print("data:\n${data}")
}
