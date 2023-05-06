import csv
import ffmpeg
import os
from tqdm import tqdm


def convert_to(input_file, output_file):
    stream = ffmpeg.input(input_file)
    stream = ffmpeg.output(stream, output_file)
    ffmpeg.run(stream)


def convert_from_tsv(path, tsv_path, output_path, output_name = "train.tsv", clip_dir = "clips/"):

    os.makedirs(output_path + clip_dir, exist_ok=True)
    with open(path + tsv_path, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        with open(output_path + output_name, 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            for row in tqdm(reader):
                filepath = row[1]
                new_path = filepath.replace(".mp3", ".wav")
                full_fp = path + clip_dir + filepath
                if os.path.exists(full_fp):
                    convert_to(full_fp, output_path + clip_dir + new_path)
                    row[1] = new_path
                    writer.writerow(row)
                elif filepath == "path":
                    # write the first row
                    writer.writerow(row)

    

if __name__ == "__main__":
    path = "C:/Downloads/cv-corpus-6.1-2020-12-11/en/"
    tsv_path = "train.tsv"

    output_path = "C:/Projects/common_voice/"

    convert_from_tsv(path, tsv_path, output_path)