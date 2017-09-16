import os
import glob
import xml.etree.ElementTree
import random
import shutil
import collections

# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

from medicalutils import medicalutils

data_dir = "/media/piotr/CE58632058630695/_dane-mgr"
dist_dir = "/media/piotr/CE58632058630695/data-sorted-5"

file_stats = collections.Counter()

validate_ratio = 0.3
shouldMergeLabels = True #whenever should use 5 (27-30 24-26 19-23 11-18  0-10 groups) or full mmse score (0 to 30)



def convertScoreToLabel(score):
    global shouldMergeLabels
    if shouldMergeLabels:
        return medicalutils.getSicknessLvl(score)['label']
    else:
        return score

def copyNiiFilesForSubject(subject, score, timestamps):
    copied_files_count = 0
    for timestamp in timestamps:
        file_dir = os.path.join(data_dir, subject, '*', timestamp, '**/*.nii')
        print(file_dir)
        for file_path in glob.iglob(file_dir, recursive=True):
            filename = os.path.basename(file_path)

            dist_image_dir = os.path.join(dist_dir, 'train', score)

            if not os.path.exists(dist_image_dir):
                os.makedirs(dist_image_dir)

            dist_image_path = os.path.join(dist_image_dir, filename)

            #print ("Copying %s to %s" % (file_path, dist_image_path))

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

def timestampToStr(timestamp):
    # 2006-11-30T05:29:33.0
    # 2006-04-18_08_20_30.0
    return timestamp.replace('T', '_').replace(':', '_').replace(' ', '_')

def loadXmlMetadata(data_dir):
    labels = set()
    dirs = glob.glob(data_dir)
    xml_file_count = len(dirs)
    xml_current_file = 0
    for xml_file in dirs:
        xml_current_file += 1
        print("Parsing %d/%d" % (xml_current_file, xml_file_count))

        root_element = xml.etree.ElementTree.parse(xml_file).getroot()
        score = None
        subject = None
        for subject_identifier in root_element.iter('subjectIdentifier'):
            subject = subject_identifier.text
            break
        for assessment in root_element.findall(".//assessmentScore[@attribute='MMSCORE']"):
            score = assessment.text
            break
        if score is not None:
            score = convertScoreToLabel(score)
            
        timestamps = []
        for timestamp in root_element.findall(".//dateAcquired"):
            timestamps.append(timestampToStr(timestamp.text))

        if subject is not None and score is not None:
            if copyNiiFilesForSubject(subject, score, timestamps) > 0:
                labels.add(score)

    saveLables(sorted(labels))

    moveValidationData()

if __name__ == '__main__':
    src_dir = data_dir + '/*.xml'
    loadXmlMetadata(src_dir)
