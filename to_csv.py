import os
import pandas as pd

data_dir = 'meta'

ids = []
images = []
labels = []

for label in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label)

    if os.path.isdir(label_path):
        for img in os.listdir(label_path):
            img_path = os.path.join(label_path, img)

            if os.path.isfile(img_path):
                ids.append(len(ids) + 1) #ids are 1 indexed
                images.append(img_path)
                labels.append(label)

data = {'id': ids, 'image': images, 'label' : labels}
df = pd.DataFrame(data)
csv_path = 'image_labels.csv'
df.to_csv(csv_path, index = False)
print(f'csv loaded, row count: {len(df)}')

                    