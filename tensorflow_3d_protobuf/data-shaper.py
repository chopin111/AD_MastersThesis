import os
import glob
import xml.etree.ElementTree
import random
import shutil
import collections

data_dir = "./data-raw"
dist_dir = "./data"

file_stats = collections.Counter()

validate_ratio = 0.3

def copyNiiFilesForSubject(subject, score):
    copied_files_count = 0
    file_dir = os.path.join(data_dir, subject, '**/*.nii')
    for file_path in glob.iglob(file_dir, recursive=True):
        filename = os.path.basename(file_path)

        dist_image_dir = os.path.join(dist_dir, 'train', score)

        if not os.path.exists(dist_image_dir):
            os.makedirs(dist_image_dir)

        dist_image_path = os.path.join(dist_image_dir, filename)

        print ("Copying %s to %s" % (file_path, dist_image_path))

        file_stats[score] += 1

        shutil.copyfile(file_path, dist_image_path)
        copied_files_count += 1
    return copied_files_count

def moveValidationData():
    print(file_stats)
    for score, count in file_stats.items():
        no_of_files_to_move = max(int(float(count)*validate_ratio), 1)

        src_filepath = os.path.join(dist_dir, 'train', score, '*.nii')

        #print (src_filepath, no_of_files_to_move)
        for filepath in glob.iglob(src_filepath, recursive=True):
            if no_of_files_to_move <= 0:
                break

            dist_image_dir = os.path.join(dist_dir, 'validate', score)

            print (dist_image_dir)

            if not os.path.exists(dist_image_dir):
                os.makedirs(dist_image_dir)

            filename = os.path.basename(filepath)

            dist_image_path = os.path.join(dist_image_dir, filename)

            print ("Moving %s to %s" % (filepath, dist_image_path))

            shutil.move(filepath, dist_image_path)

            no_of_files_to_move = no_of_files_to_move - 1

def saveLables(labels):
    with open(os.path.join(dist_dir, 'labels.txt'), 'w') as labels_file:
        for label in labels:
            labels_file.write("%s\n" % (label))

def loadXmlMetadata(data_dir):
    labels = set()
    for xml_file in glob.glob(data_dir):
        root_element = xml.etree.ElementTree.parse(xml_file).getroot()
        score = None
        subject = None
        for subject_identifier in root_element.iter('subjectIdentifier'):
            subject = subject_identifier.text
            break
        for assessment in root_element.findall(".//assessmentScore[@attribute='MMSCORE']"):
            score = assessment.text
            break

        if subject is not None and score is not None:
            if copyNiiFilesForSubject(subject, score) > 0:
                labels.add(score)

    saveLables(sorted(labels))

    moveValidationData()

if __name__ == '__main__':
    src_dir = data_dir + '/*.xml'
    loadXmlMetadata(src_dir)
