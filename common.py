import numpy as np


def random_crop(imgs: np.array, out=84) -> np.array:
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def center_crop_images(images, output_size):
    h, w = images.shape[-2:]
    if h > output_size: #center cropping
        new_h, new_w = output_size, output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        images = images[..., top:top + new_h, left:left + new_w]
        return images
    else: #center translate
        new_images = np.zeros((images.shape[0], images.shape[1], output_size, output_size))
        shift = output_size - h
        shift = shift // 2
        new_images[..., shift:shift + h, shift:shift+w] = images
        return new_images
