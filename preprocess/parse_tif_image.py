from contextlib import ExitStack
import numpy as np
from osgeo import gdal, gdalconst
from icecream import ic


def parse_tif_image(tiff_fpath):
    dataset = gdal.Open(tiff_fpath, gdal.GA_ReadOnly)
    img = dataset.ReadAsArray()
    assert (len(img.shape) == 3 and img.shape[0] == 3)
    img = img.transpose((1, 2, 0))   # [c, h, w] --> [h, w, c]
    assert (img.dtype == np.uint8)

    metadata = dataset.GetMetadata()
    date_time = metadata['NITF_IDATIM']
    year = int(date_time[0:4])
    month = int(date_time[4:6])
    day = int(date_time[6:8])
    hour = int(date_time[8:10])
    minute = int(date_time[10:12])
    second = int(date_time[12:14])
    capture_date = [year, month, day, hour, minute, second]

    rpc_data = dataset.GetMetadata('RPC')
    rpc_dict = {
        'lonOff': float(rpc_data['LONG_OFF']),
        'lonScale': float(rpc_data['LONG_SCALE']),
        'latOff': float(rpc_data['LAT_OFF']),
        'latScale': float(rpc_data['LAT_SCALE']),
        'altOff': float(rpc_data['HEIGHT_OFF']),
        'altScale': float(rpc_data['HEIGHT_SCALE']),
        'rowOff': float(rpc_data['LINE_OFF']),
        'rowScale': float(rpc_data['LINE_SCALE']),
        'colOff': float(rpc_data['SAMP_OFF']),
        'colScale': float(rpc_data['SAMP_SCALE']),
        'rowNum': np.asarray(rpc_data['LINE_NUM_COEFF'].split(), dtype=np.float64).tolist(),
        'rowDen': np.asarray(rpc_data['LINE_DEN_COEFF'].split(), dtype=np.float64).tolist(),
        'colNum': np.asarray(rpc_data['SAMP_NUM_COEFF'].split(), dtype=np.float64).tolist(),
        'colDen': np.asarray(rpc_data['SAMP_DEN_COEFF'].split(), dtype=np.float64).tolist()
    }

    meta_dict = { 'rpc': rpc_dict,
                  'height': img.shape[0],
                  'width': img.shape[1], 
                  'capture_date': capture_date
    }

    return img, meta_dict


def center_crop_tif_image(in_tiff_fpath, out_tiff_fpath, trgt_h, trgt_w): 
    in_dst = gdal.Open(in_tiff_fpath, gdal.GA_ReadOnly)
    rpc_data = in_dst.GetMetadata('RPC')

    img = in_dst.ReadAsArray()
    assert (len(img.shape) == 3 and img.shape[0] == 3)
    h, w = img.shape[1:]
    assert (h >= trgt_h and w >= trgt_w)

    ul_r = h // 2 - trgt_h // 2
    ul_c = w // 2 - trgt_w // 2
    img = img[:, ul_r:ul_r+trgt_h, ul_c:ul_c+trgt_w]
    rpc_data['LINE_OFF'] = str(float(rpc_data['LINE_OFF']) - ul_r)
    rpc_data['SAMP_OFF'] = str(float(rpc_data['SAMP_OFF']) - ul_c)

    geotiff_drv = gdal.GetDriverByName('GTiff')
    out_dst = geotiff_drv.Create(out_tiff_fpath, trgt_w, trgt_h, 3, gdalconst.GDT_Byte)
    out_dst.SetMetadata(rpc_data, 'RPC')
    for x in in_dst.GetMetadataDomainList():
        if x != 'RPC':
            out_dst.SetMetadata(in_dst.GetMetadata(x), x)

    for i in range(3):
        band = out_dst.GetRasterBand(i+1)
        band.WriteArray(img[i])
        band.FlushCache()


if __name__ == '__main__':
    # tiff_fpath = '../examples/dfc_data/inputs/JAX_167_001_RGB.tif'
    # img, meta_dict = parse_tif_image(tiff_fpath)
    # ic(type(img))
    # ic(img.shape)
    # ic(meta_dict)

    in_folder = '../examples/inputs/images'
    out_folder = '../examples/inputs/images_crop'
    import os
    os.makedirs(out_folder, exist_ok=True)
    for item in os.listdir(in_folder):
        if item.endswith('.tif'):
            center_crop_tif_image(os.path.join(in_folder, item), os.path.join(out_folder, item),
                                  trgt_h=1024, trgt_w=1024)