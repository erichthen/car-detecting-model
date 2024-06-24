import os
import pandas as pd
import shutil

#take the data from the meta directory and make a new directory
#that only contains the first 200 classes with all of their
#respective images
def refine_data(data_dir, new_dir, limit=333):
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    
    count = 0
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        
        if os.path.isdir(label_path) and count < limit:
            new_label_path = os.path.join(new_dir, label)
            os.mkdir(new_label_path)
            count += 1
            
            for img in os.listdir(label_path):
                img_path = os.path.join(label_path, img)
                
                if os.path.isfile(img_path):
                    new_img_path = os.path.join(new_label_path, img)
                    shutil.copy(img_path, new_img_path)


        else: 
            break

                    
    print(f'{limit} classes copied to {new_dir}')


if __name__ == '__main__':
    refine_data('meta', 'very_small_meta')