import os
import numpy as np
import pandas as pd


def get_landcover_train_val_df(root_path, val_fraction=0.1, random_state=None):
    ''' split train dataset into train and validate, return df of both sets.
        returned df have 2 columns, sat_image_path and mask_path (possibly image_id)
    '''
    metadata_df = pd.read_csv(os.path.join(root_path, 'metadata.csv'))
    metadata_df = metadata_df[metadata_df['split'] == 'train']
    metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]

    metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(lambda img_pth: os.path.join(root_path, img_pth))
    metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_pth: os.path.join(root_path, img_pth))
    # Shuffle DataFrame
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

    # Perform 90/10 split for train / val
    valid_df = metadata_df.sample(frac=val_fraction, random_state=random_state)
    train_df = metadata_df.drop(valid_df.index)

    return train_df, valid_df


def get_landcover_info(root_path, verbose=False):
    class_dict = pd.read_csv(os.path.join(root_path, 'class_dict.csv'))
    # Get class names
    class_names = class_dict['name'].tolist()
    # Get class RGB values
    class_rgb_values = class_dict[['r','g','b']].values.tolist()

    if verbose:
        print('All dataset classes and their corresponding RGB values in labels:')
        print('Class Names: ', class_names)
        print('Class RGB values: ', class_rgb_values)

        # Useful to shortlist specific classes in datasets with large number of classes
    select_classes = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']

    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

    if verbose:
        print('Selected classes and their corresponding RGB values in labels:')
        print('Class Names: ', class_names)
        print('Class RGB values: ', class_rgb_values)

    return {
        'class_names': class_names,
        'class_rgb_values': class_rgb_values,
        'select_class_rgb_values': select_class_rgb_values
    }
