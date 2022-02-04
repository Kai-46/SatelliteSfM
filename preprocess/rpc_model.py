def _apply_poly(poly, x, y, z):
    """
    Evaluates a 3-variables polynom of degree 3 on a triplet of numbers.

    Args:
        poly: list of the 20 coefficients of the 3-variate degree 3 polynom,
            ordered following the RPC convention.
        x, y, z: triplet of floats. They may be numpy arrays of same length.

    Returns:
        the value(s) of the polynom on the input point(s).
    """
    out = 0
    out += poly[0]
    out += poly[1]*y + poly[2]*x + poly[3]*z
    out += poly[4]*y*x + poly[5]*y*z +poly[6]*x*z
    out += poly[7]*y*y + poly[8]*x*x + poly[9]*z*z
    out += poly[10]*x*y*z
    out += poly[11]*y*y*y
    out += poly[12]*y*x*x + poly[13]*y*z*z + poly[14]*y*y*x
    out += poly[15]*x*x*x
    out += poly[16]*x*z*z + poly[17]*y*y*z + poly[18]*x*x*z
    out += poly[19]*z*z*z
    return out


class RPCModel(object):
    def __init__(self, meta_dict):
        rpc_dict = meta_dict['rpc']
        self.rowOff = rpc_dict['rowOff']
        self.rowScale = rpc_dict['rowScale']

        self.colOff = rpc_dict['colOff']
        self.colScale = rpc_dict['colScale']

        self.latOff = rpc_dict['latOff']
        self.latScale = rpc_dict['latScale']

        self.lonOff = rpc_dict['lonOff']
        self.lonScale = rpc_dict['lonScale']

        self.altOff = rpc_dict['altOff']
        self.altScale = rpc_dict['altScale']

        self.rowNum = rpc_dict['rowNum']
        self.rowDen = rpc_dict['rowDen']
        self.colNum = rpc_dict['colNum']
        self.colDen = rpc_dict['colDen']

        self.width = meta_dict['width']
        self.height = meta_dict['height']

    def projection(self, lat, lon, alt):
        cLon = (lon - self.lonOff) / self.lonScale
        cLat = (lat - self.latOff) / self.latScale
        cAlt = (alt - self.altOff) / self.altScale
    
        cCol = _apply_poly(self.colNum, cLat, cLon, cAlt) \
                / _apply_poly(self.colDen, cLat, cLon, cAlt)
        cRow = _apply_poly(self.rowNum, cLat, cLon, cAlt) \
                / _apply_poly(self.rowDen, cLat, cLon, cAlt)
    
        col = cCol*self.colScale + self.colOff
        row = cRow*self.rowScale + self.rowOff
        return col, row


if __name__ == '__main__':
    pass
