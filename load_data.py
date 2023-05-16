import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import os
import typing
import glob
import custom_dataset_script


def load_dataset(name: str, anno_path : str, img_path : str, load_split : bool = False, split_path : str = '', extension : str = 'jpg') -> fo.Dataset:
    ''' Loads VOC dataset and returns a dataset object. Need to specify directory containing
    data in VOC format, along with annotation path and image set path.  
    Can be used to load splits when the file is of the following format:
    2011_003184
    2011_003187
    2011_003188
    2011_003192
    2011_003194
    2011_003216
    2011_003223    
    .'''
    if not load_split:
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.VOCDetectionDataset,
            data_path=img_path,
            labels_path=anno_path,
            name=name
        )
        return dataset
    else:
        filenames = [f'{img_path}/{i.strip()}.{extension}' for i in open(split_path, 'r').readlines()]
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.VOCDetectionDataset,
            data_path=img_path,
            labels_path=anno_path,
            name=name
        )
        # dataset.filter_field()
        # samples = []
        for sample in dataset:
            if (sample.filepath not in filenames):
                dataset.delete_samples(sample)
                # samples.append(samples)

        return dataset
    

def load_dataset_subset(name: str, anno_path : str, img_path : str, split_path : str = '', extension : str = 'jpg') -> fo.Dataset:
    ''' Loads VOC dataset and returns a dataset object. Need to specify directory containing
    data in VOC format, along with annotation path and image set path.  
    Can be used to load splits when the file is of the following format:
    2011_003184 -1
    2011_003187  1
    2011_003188  2
    2011_003192 -1
    2011_003194  2  
    .'''
    # valid_files = []
    # for i in open(split_path, 'r').readlines():
    #     if i.strip().split(" ")[-1] != "-1":
    #         valid_files.append(i.strip().split(" ")[0])
    filenames = [f'{img_path}/{i.strip().split(" ")[0]}.{extension}' for i in open(split_path, 'r').readlines() if i.strip().split(" ")[-1] != "-1"]
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.VOCDetectionDataset,
        data_path=img_path,
        labels_path=anno_path,
        name=name
    )

    cnt = 0;
    for sample in dataset:
        if (sample.filepath not in filenames):
            dataset.delete_samples(sample)
    return dataset

def convert_dataset(dataset : fo.Dataset, label_field : str, export_dir : str, dataset_type : fo.types = fo.types.COCODetectionDataset) -> None:
    '''
    Converts any fiftyone dataset to COCODetection format. Infact, you can change the target format by changing the type argument
    '''
    # Export the dataset
    dataset.export(
        export_dir=export_dir,
        dataset_type=dataset_type,
        label_field=label_field,
    )
    print("Converted Successfully")

def custom_dataset(converted_dir : str, test_split : float, output_json : str ,bbox_format : str = 'xywh') -> None:
    '''
    Basically a wrapper around this command.
    !python custom_dataset_script.py --train_images voxelExported51/data/ --test_split 0.1 --train_labels voxelExported51/labels.json --bbox_source_format xywh -s exp_conv.json
    '''
    custom_dataset_script.build_detection_dataset_json(f'{converted_dir}/data/', f'{converted_dir}/labels.json', test_split=test_split, bbox_source_format=bbox_format, save_name=output_json)


def train_coco(batch_size : int, input_shape : int, data_name : str, lr_decay_steps : int, lr_cooldown_steps : int, freeze_backbone_epochs : int):
    '''
    Wrapper around coco_train_script:
    !python3 ./coco_train_script.py -p adamw -b 8 -i 512 --data_name exp_conv.json --lr_decay_steps 20 --lr_cooldown_steps 1 --freeze_backbone_epochs 8
    '''
    os.system(f'python3 ./coco_train_script.py -p adamw -b {batch_size} -i {input_shape} --data_name {data_name} --lr_decay_steps {lr_decay_steps} --lr_cooldown_steps {lr_cooldown_steps} --freeze_backbone_epochs {freeze_backbone_epochs}')



if __name__ == '__main__':
    # broken_dataset = load_dataset("bloodcells", "BCCD/Annotations", "BCCD/JPEGImages", True, "BCCD/ImageSets/Main/train.txt")
    # convert_dataset(broken_dataset, 'ground_truth', 'exported')
    # sesh = fo.launch_app(broken_dataset)
    ds = load_dataset_subset('dogiss', '/home/juggernautjha/Desktop/Msense/complete_training_pipeline/Datasets/VOC2012/Annotations', '/home/juggernautjha/Desktop/Msense/complete_training_pipeline/Datasets/VOC2012/JPEGImages', '../Datasets/VOC2012/ImageSets/Main/dog_train.txt')
    sesh = fo.launch_app(ds)
    sesh.wait(-1)
