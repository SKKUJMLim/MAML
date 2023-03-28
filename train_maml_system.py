from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotClassifier
from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset

from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support() # 윈도우에서 파이썬이 자원을 효율적으로 사용하게 만듬

    # Combines the arguments, model, data and experiment builders to run an experiment
    args, device = get_args()
    model = MAMLFewShotClassifier(args=args, device=device,
                                  im_shape=(2, 3,
                                            args.image_height, args.image_width))
    maybe_unzip_dataset(args=args)
    data = MetaLearningSystemDataLoader
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
    maml_system.run_experiment()

