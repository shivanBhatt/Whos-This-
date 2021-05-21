import ffmpeg
import glob
from natsort import natsorted
import os

File = open(r"output/input.txt","w+")
for frame in natsorted(os.listdir('output/')):
    if frame.endswith('.mp4'):
        File.write("file '"+frame+"'\n")
File.close()        

os.system('cmd /c "cd output & ffmpeg -f concat -i input.txt -c copy ../out.mp4"')


