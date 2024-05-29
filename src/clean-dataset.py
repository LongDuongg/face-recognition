import os
import json


def labelme_to_yolo_box(top_left, bottom_right, img_width, img_height):
    x_min = top_left[0]
    y_min = top_left[1]
    x_max = bottom_right[0]
    y_max = bottom_right[1]

    width = x_max - x_min
    height = y_max - y_min

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # YOLO normalises the image space to run from 0 to 1 in both x and y directions
    x_center = x_center / img_width
    y_center = y_center / img_height
    width = width / img_width
    height = height / img_height

    return (x_center, y_center, width, height)


def convert(input_folder_path, output_folder_path):
    for json_file_name in os.listdir(input_folder_path):
        json_file_path = input_folder_path + "/" + json_file_name
        txt_file_name = json_file_name.rstrip(".json") + ".txt"
        txt_file_path = output_folder_path + "/" + txt_file_name

        json_outfile = open(json_file_path)
        data = json.load(json_outfile)

        shape = data["shapes"][0]

        yolo_box = labelme_to_yolo_box(
            shape["points"][0],
            shape["points"][1],
            data["imageWidth"],
            data["imageHeight"],
        )

        txt_data = shape["label"] + " " + " ".join([str(v) for v in yolo_box]) + "\n"

        txt_outfile = open(txt_file_path, "w")
        txt_outfile.write(txt_data)
        txt_outfile.close()


def run():
    # dp = "./dataset/test"
    # dp = "./dataset/train"
    dp = "./dataset/valid"

    classes = ["Long", "Phuc", "Quoc"]

    for cls in classes:
        # ./dataset/test/Long_Label => ./dataset/test/long-labels
        input_folder_path = dp + "/" + cls + "_" + "Label"
        output_folder_path = dp + "/" + cls.lower() + "-" + "labels"
        os.mkdir(output_folder_path)
        print(input_folder_path, output_folder_path)
        convert(input_folder_path, output_folder_path)


def rename():
    # dp = "./dataset/test"
    # dp = "./dataset/train"
    dp = "./dataset/valid"

    classes = ["Long", "Phuc", "Quoc"]

    def folder_names(cls):
        txt_folder_name = cls.lower() + "-" + "labels"
        json_folder_name = cls + "_" + "Label"
        return (cls, txt_folder_name, json_folder_name)

    for cls in classes:
        for folder_name in folder_names(cls):
            for file_name in os.listdir(os.path.join(dp, folder_name)):
                new_file_name = cls.lower() + "_" + file_name
                os.rename(
                    os.path.join(dp, folder_name, file_name),
                    os.path.join(dp, folder_name, new_file_name),
                )


# run()
rename()
