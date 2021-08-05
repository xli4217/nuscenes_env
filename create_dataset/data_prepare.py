import os
import fire
import pandas as pd
from create_dataset.ray_data_processor import RayDataProcessor
from create_dataset.data_prepare_configs import get_config
from create_dataset.process_normalize_and_split import ProcessDatasetSplit
from pathlib import Path

def create(mode='raw', num_workers=1, dataset_type='mini', test=False):

    config = get_config(dataset_type=dataset_type,
                        data_root_dir=os.path.join(str(Path(os.environ['PKG_PATH']).parent), 'data_df'),
                        num_workers=num_workers,
                        mode=mode,
                        test=test
)
    if mode != 'final':
        cls = RayDataProcessor(config)
        cls.run()
    elif mode == 'final':
        processor = ProcessDatasetSplit(config=config)
        df = processor.process()
    else:
        raise ValueError('mode not supported')
        
if __name__ == "__main__":
    fire.Fire(create)
