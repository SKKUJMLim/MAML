from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder

from few_shot_learning_system import MAMLFewShotClassifier
# from few_shot_learning_system_LSTMArbiter import MAMLFewShotClassifier

from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset

from multiprocessing import freeze_support

''' Prompter experiment'''
## python train_maml_system.py --name_of_args_json_file experiment_config/prompt_config/MAML+Arbiter_5way_5shot_filter48.json --gpu_to_use 0

''' MAML experiment'''
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML_5way_5shot_filter48.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML_5way_5shot_filter64.json --gpu_to_use 0

## python train_maml_system.py --name_of_args_json_file experiment_config/MAML_5way_1shot.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML_5way_5shot.json --gpu_to_use 0

## python train_maml_system.py --name_of_args_json_file experiment_config/MAML_tired_imagenet_5way_1shot_filter128.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML_tired_imagenet_5way_5shot_filter128.json --gpu_to_use 0

'''GAP experiment'''
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+GAP_5way_5shot.json --gpu_to_use 0

''' Resnet experiment'''
# 1) mini imagenet
## python train_maml_system.py --name_of_args_json_file experiment_config/resnet/miniimagenet/MAML+Arbiter_5way_1shot_resnet.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/resnet/miniimagenet/MAML+Arbiter_5way_5shot_resnet.json --gpu_to_use 0
# 2) tiered imagenet
## python train_maml_system.py --name_of_args_json_file experiment_config/resnet/tiredimagenet/MAML+Arbiter_tired_imagenet_5way_1shot_resnet.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/resnet/tiredimagenet/MAML+Arbiter_tired_imagenet_5way_5shot_resnet.json --gpu_to_use 0


''' VGG experiment'''
# 1) mini imagenet
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Arbiter_5way_1shot.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Arbiter_5way_5shot.json --gpu_to_use 0

## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Arbiter_5way_5shot_filter32.json --gpu_to_use 0

## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Arbiter_5way_1shot_filter48.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Arbiter_5way_5shot_filter48.json --gpu_to_use 0

## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Arbiter_5way_1shot_filter64.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Arbiter_5way_5shot_filter64.json --gpu_to_use 0

# 2) tiered imagenet
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Arbiter_tiered_imagenet_5way_1shot.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Arbiter_tiered_imagenet_5way_5shot.json --gpu_to_use 0

## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Arbiter_tired_imagenet_5way_1shot_filter48.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Arbiter_tired_imagenet_5way_5shot_filter48.json --gpu_to_use 0

## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Arbiter_tired_imagenet_5way_1shot_filter64.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Arbiter_tired_imagenet_5way_5shot_filter64.json --gpu_to_use 0


if __name__ == '__main__':
    freeze_support()

    # Combines the arguments, model, data and experiment builders to run an experiment
    args, device = get_args()

    # 모델을 구성한다
    model = MAMLFewShotClassifier(args=args, device=device,
                                  im_shape=(2, 3,
                                            args.image_height, args.image_width))
    maybe_unzip_dataset(args=args)

    # 데이터를 불러온다
    data = MetaLearningSystemDataLoader

    # 학습
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
    maml_system.run_experiment()

