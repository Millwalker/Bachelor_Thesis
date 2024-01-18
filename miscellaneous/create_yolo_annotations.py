import os
import re

dir_str = '/home/julian/Bachelor Thesis/box_2d_annotations' 
targ_dir_str = '/home/julian/Bachelor Thesis/box_2d_annotations_new' 

directory = os.fsencode(dir_str)
targ_directory = targ_dir_str

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)

w = 1280
h = 966
for file in os.listdir(directory):

    file_name = os.fsdecode(file)
    file_name = '/' + file_name
    file_name = dir_str + file_name[0:]

    targ_file = targ_directory
    targ_file += '/'
    targ_file += file_name[-13:]

    with open(file_name) as infile, open(targ_file, 'w') as outfile:
        for line in infile:
            line = line.replace(',', ' ')
            line = line.replace('person ', '')

            if line.startswith('1'):
                numbers = re.findall(r'\d+', line)
                numbers = list(map(int, numbers))
                b = (numbers[1], numbers[3], numbers[2], numbers[4])
                bb = convert((w, h), b)
                # Writing in YOLO format where class is 0 (person)
                outfile.write("%s %s %s %s %s\n" % (0, bb[0], bb[1], bb[2], bb[3]))

print('Done')

