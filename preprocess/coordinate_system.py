import numpy as np
import pymap3d
import utm
import pyproj


def latlonalt_to_enu(lat, lon, alt, observer_lat, observer_lon, observer_alt):
    '''

    '''
    e, n, u = pymap3d.geodetic2enu(lat, lon, alt, observer_lat, observer_lon, observer_alt)

    return e, n, u


def enu_to_latlonalt(e, n, u, observer_lat, observer_lon, observer_alt):
    lat, lon, alt = pymap3d.enu2geodetic(e, n, u, observer_lat, observer_lon, observer_alt)

    return lat, lon, alt


def latlon_to_utm(lat, lon):
    '''
        lat: [N, ]; numpy array
        lon: [N, ]; numpy array

        return:
            utm_e: [N, ]; numpy array
            utm_n: [N, ]; numpy array
    '''
    assert(np.all(lat >= 0) or np.all(lat < 0))

    is_south_hemi = False if lat[0] >= 0 else True
    _, _, zone_number, _ = utm.from_latlon(lat[0], lon[0])

    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=zone_number, south=is_south_hemi)
    utm_e, utm_n = proj(lon, lat)
    return utm_e, utm_n


def utm_to_latlon(utm_e, utm_n, zone_number, is_north_hemi):
    '''
        utm_e: [N, ]; numpy array
        utm_n: [N, ]; numpy array
        zone_number: integer
        is_north_hemi: true or false

        return:
            lat: [N, ]; numpy array
            lon: [N, ]; numpy array
    '''
    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=zone_number, south=not is_north_hemi)
    lon, lat = proj(utm_e, utm_n, inverse=True)
    return lat, lon
