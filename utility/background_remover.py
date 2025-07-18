import os
from PIL import Image

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert("RGBA")
            datas = img.getdata()

            new_data = []
            for item in datas:
                # Change all white (also shades of whites)
                # to transparent
                if item[0] >= 240 and item[1] >= 240 and item[2] >= 240:
                    new_data.append((255, 255, 255, 0))  # Set white to transparent
                else:
                    new_data.append(item)

            img.putdata(new_data)

            # Save the new image
            new_filename = os.path.splitext(filename)[0] + "_transp.png"
            new_img_path = os.path.join(output_folder, new_filename)
            img.save(new_img_path)

input_folder = '/Users/student/Desktop/Work/Barmak_Lab/Coding_Things/grain_unet_working/Data/training_validation_data/train_nouveaux_256/label'
output_folder = '/Users/student/Desktop/Work/Barmak_Lab/Coding_Things/grain_unet_working/Data/training_validation_data/train_nouveaux_256/news'

process_images(input_folder, output_folder)
