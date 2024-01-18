import os
import shutil

def joinStrings(stringList):
    return ''.join(string for string in stringList)
    
folderdir = "remain_3"
dirs = os.listdir(folderdir)
count=0
for i in dirs:
    if i.endswith(".txt"):
        txt = i
        jpg = str(i.split(".txt")[0]) + ".jpg"
        if os.stat(joinStrings([folderdir,'/',txt])).st_size == 0:
        	count+=1
        else:
        	shutil.copy(folderdir + "/" + jpg, 'remain_img3')
        	shutil.copy(folderdir + "/" + txt, 'remain_anno3')
       
        	
        	#print("this one")
    
print("Images and Annotations not copied: " + str(count))

