import pandas as pd
import numpy as np
import cv2
import os
from skimage.io import imread
from tqdm import tqdm


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(row, all_masks=None):
    """

    :param row: takes DataFrame row
    :param all_masks:
    :return: np.array with shape (768,768,3)
    """
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype=np.uint8)

    if isinstance(row['EncodedPixels'], str):
        all_masks += rle_decode(row['EncodedPixels'])
    return np.expand_dims(all_masks, -1)


def process_dataframe(data, print_log=True):
    """
    :param data: DataFrame with RLE and Image Path
    :param print_log: if you want to print memory usage of created np.arrays
    :return: dict {masks: np.array(num_rows, 768,768,1), "images_rgb": (num_rows, 768,768,3)}
    """
    num_rows = len(data)
    masks = np.zeros((num_rows, 768, 768, 1), dtype=np.uint8)
    images_rgb = np.zeros((num_rows, 768, 768, 3), dtype=np.uint8)

    for i in range(num_rows):
        mask = masks_as_image(data.iloc[i])
        rgb = imread(os.path.join('data/train_v2', data.iloc[i]['ImageId']))

        masks[i, :, :, :] = mask
        images_rgb[i, :, :, :] = rgb
    result_dict = {'masks': masks, 'images_rgb': images_rgb}
    if print_log:
        get_memory_usage(result_dict, f'Original size {(768, 768)}')

    return result_dict


def resize_image(image, size=(256, 256)):
    """

    :param image: np.array representation of image
    :param size: size you want to compress your original image
    :return: resized np.array
    """
    return cv2.resize(image, size)


def resize_mask(mask, size=(256, 256)):
    """
    :param mask: np.array representation of mask
    :param size: size you want to compress your original mask
    :return: resized np.array
    """
    return cv2.resize(mask, size)


def rle_decode_resized(mask_rle, original_shape=(768, 768), resized_shape=(256, 256)):
    """

    :param mask_rle: RLE code
    :param original_shape:
    :param resized_shape:
    :return: np.array with resized shape
    """
    img = rle_decode(mask_rle, original_shape)
    resized_img = resize_mask(img, resized_shape)
    return resized_img


def masks_as_image_resized(row, all_masks=None, resized_shape=(256, 256)):
    """

    :param row:
    :param all_masks:
    :param resized_shape:
    :return:
    """
    if all_masks is None:
        all_masks = np.zeros(resized_shape, dtype=np.uint8)

    if isinstance(row['EncodedPixels'], str):
        mask = rle_decode_resized(row['EncodedPixels'])
        all_masks += mask

    return np.expand_dims(all_masks, -1)


def process_dataframe_resized(data, target_size=(256, 256), print_log=True):
    num_rows = len(data)
    masks = np.zeros((num_rows, *target_size, 1), dtype=np.uint8)
    images_rgb = np.zeros((num_rows, *target_size, 3), dtype=np.uint8)

    for i in range(num_rows):
        mask = masks_as_image_resized(data.iloc[i], resized_shape=target_size)
        rgb = imread(os.path.join('data/train_v2', data.iloc[i]['ImageId']))
        rgb_resized = resize_image(rgb, target_size)

        masks[i, :, :, :] = mask
        images_rgb[i, :, :, :] = rgb_resized

    result_dict = {'masks': masks, 'images_rgb': images_rgb}
    if print_log:
        get_memory_usage(result_dict, f'Resized to {target_size}')

    return result_dict


def get_data():
    df = pd.read_csv('data/train_ship_segmentations_v2.csv')
    df = df[~df['ImageId'].isin(['6384c3e78.jpg'])]
    print('Total rows:', len(df))
    df_with_ships = df[~df['EncodedPixels'].isna()]
    print('Total ships on images:', len(df_with_ships))
    df_without_ships = df[df['EncodedPixels'].isna()]
    print('Total images without ships:', len(df_without_ships))
    df_with_ships = df_with_ships.groupby("ImageId")[['EncodedPixels']].agg(
        lambda rle_codes: ' '.join(rle_codes)).reset_index()
    print('Total images with ships:', len(df_with_ships))
    df_without_ships = df_without_ships.sample(n=20000, random_state=42)

    data = pd.concat([df_with_ships, df_without_ships], ignore_index=True)

    print('Under sampled data with 20k non ships images + images with ships has:', len(data))
    return data


def get_memory_usage(preprocessed_data: dict, type_data=''):
    print(f"{type_data} : Masks memory usage: {preprocessed_data['masks'].nbytes // (1024 ** 2)} Mb")
    print(f"{type_data} : Image memory usage: {preprocessed_data['images_rgb'].nbytes // (1024 ** 2)} Mb")


def get_preprocessed_batches(data, batch_size=64, print_log=False, resized=True):
    all_indices = data.index.tolist()

    np.random.shuffle(all_indices)
    batches = [all_indices[i:i + batch_size] for i in range(0, len(all_indices), batch_size)]

    batches_data = {}

    i = 0
    print(f'Process all data as batches with size {batch_size}')
    for batch_indices in tqdm(batches):
        batch_data = data.loc[batch_indices]
        if resized:
            temp_dict = process_dataframe_resized(batch_data, print_log=print_log)
        else:
            temp_dict = process_dataframe(batch_data, print_log=print_log)
        batches_data[i] = temp_dict
        i += 1

    return batches_data


def get_X_y_data(data, resized=False):
    if resized:
        temp_dict = get_preprocessed_batches(data, resized=True)
        masks_combined = np.empty((0, 256, 256, 1), dtype=np.float32)
        images_rgb_combined = np.empty((0, 256, 256, 3), dtype=np.float32)
    else:
        temp_dict = get_preprocessed_batches(data, resized=False)
        masks_combined = np.empty((0, 768, 768, 1), dtype=np.float32)
        images_rgb_combined = np.empty((0, 768, 768, 3), dtype=np.float32)

    print('Merge all batches into one big preprocessed DataSet')
    for key in tqdm(temp_dict):
        masks_combined = np.concatenate((masks_combined, temp_dict[key]['masks']), axis=0)
        images_rgb_combined = np.concatenate((images_rgb_combined, temp_dict[key]['images_rgb']), axis=0)

    print('X shape:', images_rgb_combined.shape)
    print('y shape:', masks_combined.shape)

    return images_rgb_combined, masks_combined

