#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import sys
import math
import argparse
from transform_coord.coordTransform_utils import gcj02_to_bd09
from transform_coord.coordTransform_utils import bd09_to_gcj02
from transform_coord.coordTransform_utils import wgs84_to_gcj02
from transform_coord.coordTransform_utils import gcj02_to_wgs84
from transform_coord.coordTransform_utils import bd09_to_wgs84
from transform_coord.coordTransform_utils import wgs84_to_bd09

# Configuration
# Input file name
INPUT = ''
# Output file name
OUTPUT = ''
# Convert type: g2b, b2g, w2g, g2w, b2w, w2b
TYPE = ''
# lng column name
LNG_COLUMN = ''
# lat column name
LAT_COLUMN = ''
# Skip invalid row
SKIP_INVALID_ROW = False


def convert():
    with open(INPUT, 'r') as input_file:
        input_file_reader = csv.reader(input_file)
        headers = next(input_file_reader)
        lng_index, lat_index = get_lng_lat_index(headers)
        results = []

        for index, row in enumerate(input_file_reader):
            result = []
            try:
                result = convert_by_type(float(row[lng_index]), float(row[lat_index]), TYPE)
            except ValueError:
                # Deal with ValueError(invalid lng or lat)
                # print(index + 2, row[lng_index], row[lat_index]) # '+ 2' is due to zero-based index and first row is header
                result = row[lng_index], row[lat_index]
            results.append(result)

    with open(OUTPUT, 'w') as output_file:
        output_file_writer = csv.writer(output_file)

        with open(INPUT, 'r') as input_file:
            input_file_reader = csv.reader(input_file)
            headers = next(input_file_reader)
            lng_index, lat_index = get_lng_lat_index(headers)

            output_file_writer.writerow(headers)
            for index, row in enumerate(input_file_reader):
                row[lng_index] = results[index][0]
                row[lat_index] = results[index][1]
                if type(row[lng_index]) is not float or type(row[lat_index]) is not float:
                    # Data is invalid
                    if SKIP_INVALID_ROW:
                        # Skip invalid row
                        pass
                    else:
                        # Reserve invalid row
                        output_file_writer.writerow(row)
                else:
                    # Data is valid
                    output_file_writer.writerow(row)


def get_lng_lat_index(headers):
    try:
        if LNG_COLUMN == '' and LAT_COLUMN == '':
            return [headers.index('lng'), headers.index('lat')]
        else:
            return [headers.index(LNG_COLUMN), headers.index(LAT_COLUMN)]
    except ValueError as error:
        print('Error: ' + str(error).split('is', 1)[0] + 'is missing from csv header. Or use -n or -a to specify custom column name for lng or lat.')
        sys.exit()


def convert_by_type(lng, lat, type):
    if type == 'g2b':
        return gcj02_to_bd09(lng, lat)
    elif type == 'b2g':
        return bd09_to_gcj02(lng, lat)
    elif type == 'w2g':
        return wgs84_to_gcj02(lng, lat)
    elif type == 'g2w':
        return gcj02_to_wgs84(lng, lat)
    elif type == 'b2w':
        return bd09_to_wgs84(lng, lat)
    elif type == 'w2b':
        return wgs84_to_bd09(lng, lat)
    else:
        print('Usage: type must be in one of g2b, b2g, w2g, g2w, b2w, w2b')
        sys.exit()


# new york standard zone is 18
def utm_to_latlng(zone, easting, northing, northernHemisphere=True):

    if not northernHemisphere:
        northing = 10000000 - northing

    a = 6378137
    e = 0.081819191
    e1sq = 0.006739497
    k0 = 0.9996

    arc = northing / k0
    mu = arc / (a * (1 - math.pow(e, 2) / 4.0 - 3 * math.pow(e, 4) / 64.0 - 5 * math.pow(e, 6) / 256.0))

    ei = (1 - math.pow((1 - e * e), (1 / 2.0))) / (1 + math.pow((1 - e * e), (1 / 2.0)))

    ca = 3 * ei / 2 - 27 * math.pow(ei, 3) / 32.0

    cb = 21 * math.pow(ei, 2) / 16 - 55 * math.pow(ei, 4) / 32
    cc = 151 * math.pow(ei, 3) / 96
    cd = 1097 * math.pow(ei, 4) / 512
    phi1 = mu + ca * math.sin(2 * mu) + cb * math.sin(4 * mu) + cc * math.sin(6 * mu) + cd * math.sin(8 * mu)

    n0 = a / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (1 / 2.0))

    r0 = a * (1 - e * e) / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (3 / 2.0))
    fact1 = n0 * math.tan(phi1) / r0

    _a1 = 500000 - easting
    dd0 = _a1 / (n0 * k0)
    fact2 = dd0 * dd0 / 2

    t0 = math.pow(math.tan(phi1), 2)
    Q0 = e1sq * math.pow(math.cos(phi1), 2)
    fact3 = (5 + 3 * t0 + 10 * Q0 - 4 * Q0 * Q0 - 9 * e1sq) * math.pow(dd0, 4) / 24

    fact4 = (61 + 90 * t0 + 298 * Q0 + 45 * t0 * t0 - 252 * e1sq - 3 * Q0 * Q0) * math.pow(dd0, 6) / 720

    lof1 = _a1 / (n0 * k0)
    lof2 = (1 + 2 * t0 + Q0) * math.pow(dd0, 3) / 6.0
    lof3 = (5 - 2 * Q0 + 28 * t0 - 3 * math.pow(Q0, 2) + 8 * e1sq + 24 * math.pow(t0, 2)) * math.pow(dd0, 5) / 120
    _a2 = (lof1 - lof2 + lof3) / math.cos(phi1)
    _a3 = _a2 * 180 / math.pi

    latitude = 180 * (phi1 - fact1 * (fact2 + fact3 + fact4)) / math.pi

    if not northernHemisphere:
        latitude = -latitude

    longitude = ((zone > 0) and (6 * zone - 183.0) or 3.0) - _a3

    return (latitude, longitude)


if __name__ == '__main__':

    # print(gcj02_to_wgs84(116.4172, 39.93889))

    print(wgs84_to_gcj02(4326, 39.93889))

    print(utm_to_latlng(18, 538090.21382165, 4436628.55154459))

    # parser = argparse.ArgumentParser(description='Convert coordinates in csv files.', usage='%(prog)s [-h] -i INPUT -o OUTPUT -t TYPE [-n LNG_COLUMN] [-a LAT_COLUMN] [-s SKIP_INVALID_ROW]', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #
    # group = parser.add_argument_group('arguments')
    #
    # group.add_argument('-i', '--input', help='Location of input file', default=argparse.SUPPRESS, metavar='')
    # group.add_argument('-o', '--output', help='Location of output file', default=argparse.SUPPRESS, metavar='')
    # group.add_argument('-t', '--type', help='Convert type, must be one of: g2b, b2g, w2g, g2w, b2w, w2b', default=argparse.SUPPRESS, metavar='')
    # group.add_argument('-n', '--lng_column', help='Column name for longitude', default='lng', metavar='')
    # group.add_argument('-a', '--lat_column', help='Column name for latitude', default='lat', metavar='')
    # group.add_argument('-s', '--skip_invalid_row', help='Whether to skip invalid row', default=False, type=bool, metavar='')
    #
    # args = parser.parse_args()
    # # print('\nArguments you provide are:')
    # # for arg in vars(args):
    # #     print '{0:20} {1}'.format(arg, str(getattr(args, arg)))
    #
    # # Get arguments
    # if not args.input or not args.output or not args.type:
    #     parser.print_help()
    # else:
    #     INPUT = args.input
    #     OUTPUT = args.output
    #     TYPE = args.type
    #
    # if args.lng_column and args.lat_column:
    #     LNG_COLUMN, LAT_COLUMN = args.lng_column, args.lat_column
    #
    # if args.skip_invalid_row:
    #     SKIP_INVALID_ROW = args.skip_invalid_row
    #
    # convert()
