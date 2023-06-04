import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import os
import typing
import glob
from tqdm import tqdm
import custom_dataset_script
import subprocess
import voc2coco
import train_coco_config
import train_with_config 
import shutil

### NOTE: This uses Voxel51
def load_dataset(anno_path : str, img_path : str, load_split : bool = False, split_path : str = '', extension : str = 'jpg') -> fo.Dataset:
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
        )
        return dataset
    else:
        filenames = [f'{img_path}/{i.strip()}.{extension}' for i in open(split_path, 'r').readlines()]
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.VOCDetectionDataset,
            data_path=img_path,
            labels_path=anno_path,
        )
        # dataset.filter_field()
        # samples = []
        temp_Df = fo.Dataset()
        for sample in dataset:
            if (sample.filepath in filenames):
                dataset.delete_samples(sample)
                # samples.append(samples)

        return dataset
    

def load_dataset_subset(anno_path : str, img_path : str, split_path : str = '', extension : str = 'jpg') -> fo.Dataset:
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
    filenames = [f'{img_path}/{i.strip().split(" ")[0]}.{extension}' for i in tqdm(open(split_path, 'r').readlines()) if i.strip().split(" ")[-1] != "-1"]
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.VOCDetectionDataset,
        data_path=img_path,
        labels_path=anno_path,
    )

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

###NOTE: This uses Voc2Coco and DOES require a path list. I'll probably write a wrapper on top of it. TODO.
def extract_labels(ann_dir : str, outfile : str):
    #! run it once to extract all labels from annotations. this is required for the next step.
    os.system(f"grep -ERoh '<name>(.*)</name>' {ann_dir} | sort | uniq | sed 's/<name>//g' | sed 's/<\/name>//g' > {outfile}")

def convert_all_using_voc2coco(ann_dir : str, img_dir : str, labels : str, output : str, extract_num : bool = False,  extension  : str = 'xml', img_extension : str = 'jpg') -> None:
    '''
    Creates a directory with the following layout
    dir_name/
        data/ -> contains the relevant images
        labels.json -> coco labels.json

    '''
    if not os.path.exists(f'{output}/data'):
        os.makedirs(f'{output}/data')
    ann_ids = os.listdir(ann_dir)
    slugs = [f'{i.split(".")[0]}' for i in ann_ids]
    img_paths = [f'{img_dir}/{i}.jpg' for i in slugs]
    dest_paths = [f'{output}/data/{i}.jpg' for i in slugs]
    for a,b in tqdm(zip(img_paths, dest_paths)):
        shutil.copy(a, b)
    label2id = voc2coco.get_label2id(labels)
    ann_paths = [f'{ann_dir}/{i}' for i in ann_ids]
    voc2coco.convert_xmls_to_cocojson(ann_paths, label2id=label2id, output_jsonpath=f'{output}/labels.json', extract_num_from_imgid=extract_num)

def convert_using_voc2coco(ann_dir : str, ann_ids : str, img_dir : str, labels : str, output : str, extract_num : bool = False,  extension  : str = 'xml', img_extension : str = 'jpg') -> None:
    '''
    Creates a directory with the following layout
    dir_name/
        data/ -> contains the relevant images
        labels.json -> coco labels.json

    '''
    if not os.path.exists(f'{output}/data'):
        os.makedirs(f'{output}/data')
    slugs = [f'{i.strip()}.jpg' for i in open(ann_ids).readlines()]
    img_paths = [f'{img_dir}/{i}' for i in slugs]
    dest_paths = [f'{output}/data/{i}' for i in slugs]
    for a,b in tqdm(zip(img_paths, dest_paths)):
        shutil.copy(a, b)
    label2id = voc2coco.get_label2id(labels)
    ann_paths = voc2coco.get_annpaths(ann_dir_path=ann_dir, ann_ids_path=ann_ids, ext=extension)
    voc2coco.convert_xmls_to_cocojson(ann_paths, label2id=label2id, output_jsonpath=f'{output}/labels.json', extract_num_from_imgid=extract_num)



###NOTE: tool Agnostic.
def custom_dataset(converted_dir : str, test_split : float, output_json : str ,bbox_format : str = 'xywh') -> None:
    '''
    Basically a wrapper around this command.
    !python custom_dataset_script.py --train_images voxelExported51/data/ --test_split 0.1 --train_labels voxelExported51/labels.json --bbox_source_format xywh -s exp_conv.json
    '''
    custom_dataset_script.build_detection_dataset_json(f'{converted_dir}/data/', f'{converted_dir}/labels.json', test_split=test_split, bbox_source_format=bbox_format, save_name=output_json)

def custom_dataset_split(train_dir : str, test_dir : str, output_json : str, bbox_format : str = 'xywh') -> None:
    custom_dataset_script.build_detection_dataset_json(f'{train_dir}/data/', f'{train_dir}/labels.json',f'{test_dir}/data/', f'{test_dir}/labels.json' , bbox_source_format=bbox_format, save_name=output_json)
    



def train_coco(batch_size : int, input_shape : int, data_name : str, lr_decay_steps : int, lr_cooldown_steps : int, freeze_backbone_epochs : int):
    '''
    Wrapper around coco_train_script:
    !python3 ./coco_train_script.py -p adamw -b 8 -i 512 --data_name exp_conv.json --lr_decay_steps 20 --lr_cooldown_steps 1 --freeze_backbone_epochs 8
    '''
    call = f'python3 ./coco_train_script.py -p adamw -b {batch_size} -i {input_shape} --data_name {data_name} --lr_decay_steps {lr_decay_steps} --lr_cooldown_steps {lr_cooldown_steps} --freeze_backbone_epochs {freeze_backbone_epochs}'
    # os.system(f'python3 ./coco_train_script.py -p adamw -b {batch_size} -i {input_shape} --data_name {data_name} --lr_decay_steps {lr_decay_steps} --lr_cooldown_steps {lr_cooldown_steps} --freeze_backbone_epochs {freeze_backbone_epochs}')
    print(subprocess.check_output(call.split(' ')))

def train_using_config(config_file : str, coco : bool = True):
    if coco:
        args = train_coco_config.parse_arguments(config_file)
        cyan_print = lambda ss: print("\033[1;36m" + ss + "\033[0m")
        if args.freeze_backbone_epochs - args.initial_epoch > 0:
            total_epochs = args.epochs
            cyan_print(">>>> Train with freezing backbone")
            args.additional_det_header_kwargs.update({"freeze_backbone": True})
            args.epochs = args.freeze_backbone_epochs
            model, latest_save, _ = train_coco_config.run_training_by_args(args)

            cyan_print(">>>> Unfreezing backbone")
            args.additional_det_header_kwargs.update({"freeze_backbone": False})
            args.initial_epoch = args.freeze_backbone_epochs
            args.epochs = total_epochs
            args.backbone_pretrained = None
            args.restore_path = None
            args.pretrained = latest_save  # Build model and load weights

        train_coco_config.run_training_by_args(args)
    
    else:
        args = train_with_config.parse_arguments(config_file)
        train_with_config.run_training_by_args(args)





if __name__ == '__main__':
    # broken_dataset = load_dataset("bloodcells", "BCCD/Annotations", "BCCD/JPEGImages", True, "BCCD/ImageSets/Main/train.txt")
    # convert_dataset(broken_dataset, 'ground_truth', 'exported')
    # sesh = fo.launch_app(broken_dataset)
    # train_using_config('config.json', True)
    # convert_all_using_voc2coco(ann_dir="/home/juggernautjha/Desktop/Msense/complete_training_pipeline/Datasets/VOC2012/Annotations", \
    #                         labels="/home/juggernautjha/Desktop/Msense/complete_training_pipeline/Datasets/VOC2012/labels.txt", \
    #                         img_dir="/home/juggernautjha/Desktop/Msense/complete_training_pipeline/Datasets/VOC2012/JPEGImages",\
    #                         output="../testing123") 
    
    custom_dataset("../testing123", 0.1, "lll.json")



    
