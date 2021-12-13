import sys
import os
import numpy as np
import cv2
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import xml.etree.ElementTree as ET
from xml.dom import minidom
import geojson
import argparse
from fastai.vision.all import *
import matplotlib.pyplot as plt
import fastai
import PIL
matplotlib.use('Agg')
import pandas as pd
import datetime
from skimage import draw, measure, morphology
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, shape
from shapely.ops import cascaded_union, unary_union
import json
import shapely
import warnings
warnings.filterwarnings("ignore")
from semtorch import get_segmentation_learner


class extractPatch:

    def __init__(self):
        self.file_location = args.file_location
        self.image_file = args.image_file
        self.save_location = args.save_dir
        self.save_name = args.save_name #'CZ2'#'trythis'
        self.mag_extract = [5] # specify which magnifications you wish to pull images from
        self.save_image_size = 500   # specify image size to be saved (note this is the same for all magnifications)
        self.pixel_overlap = 100       # specify the level of pixel overlap in your saved images
        self.limit_bounds = True     # this is weird, dont change it
        self.model_path = '/path/to/deeplabv3_resnet50_10ep_lr1e4_nonorm'

    def parseMeta_and_pullTiles(self):
        if not os.path.exists(os.path.join(self.save_location)):
            os.mkdir(os.path.join(self.save_location))

        fns = pd.read_csv('/path/to/dummy/trainval_dummy.csv')
        codes = ['Background', 'Tumor']
        segdata = DataBlock(blocks=(ImageBlock, MaskBlock), splitter=ColSplitter(), get_x=ColReader('img'),
                            get_y=ColReader('label'), item_tfms=[Resize((500, 500))],
                            batch_tfms=[Normalize.from_stats(*imagenet_stats)])
        dls = segdata.dataloaders(fns, bs=4, tfm_y=True)
        learn = get_segmentation_learner(dls=dls, number_classes=2, segmentation_type="Semantic Segmentation",
                                         architecture_name="deeplabv3+", backbone_name="resnet50")
        learn.load(self.model_path)


        #first load pytorch model
        #learn = load_learner(self.model_path,cpu=False)

        # first grab data from digital header
        # print(os.path.join(self.file_location,self.image_file))
        oslide = openslide.OpenSlide(os.path.join(self.file_location,self.image_file))

        # this is physical microns per pixel
        acq_mag = 10.0/float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])

        # this is nearest multiple of 20 for base layer
        base_mag = int(20 * round(float(acq_mag) / 20))

        # this is how much we need to resample our physical patches for uniformity across studies
        physSize = round(self.save_image_size*acq_mag/base_mag)

        # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
        tiles = DeepZoomGenerator(oslide, tile_size=physSize-round(self.pixel_overlap*acq_mag/base_mag), overlap=round(self.pixel_overlap*acq_mag/base_mag/2), limit_bounds=self.limit_bounds)

        # calculate the effective magnification at each level of tiles, determined from base magnification
        tile_lvls = tuple(base_mag/(tiles._l_z_downsamples[i]*tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for i in range(0,tiles.level_count))

        # pull tiles from levels specified by self.mag_extract
        for lvl in self.mag_extract:
            if lvl in tile_lvls:
                # print(lvl)
                # pull tile info for level
                x_tiles, y_tiles = tiles.level_tiles[tile_lvls.index(lvl)]

                # note to self, we have to iterate b/c deepzoom does not allow casting all at once at list (??)
                polygons = []
                # xy_lim = self.get_box(path=self.xml_file)
                for y in range(0,y_tiles):
                    for x in range(0,x_tiles):

                        # grab tile coordinates
                        tile_coords = tiles.get_tile_coordinates(tile_lvls.index(lvl), (x, y))
                        save_coords = str(tile_coords[0][0]) + "-" + str(tile_coords[0][1]) + "_" + '%.0f'%(tiles._l0_l_downsamples[tile_coords[1]]*tile_coords[2][0]) + "-" + '%.0f'%(tiles._l0_l_downsamples[tile_coords[1]]*tile_coords[2][1])

                        tile_pull = tiles.get_tile(tile_lvls.index(lvl), (x, y))
                        ws = self.whitespace_check(im=tile_pull)
                        if ws < 0.95:
                            tile_pull = tile_pull.resize(size=(self.save_image_size, self.save_image_size),resample=PIL.Image.ANTIALIAS)
                            tile_pull = np.array(tile_pull)
                            inp, targ, pred, _ = learn.predict(tile_pull, with_input=True)
                            pred_arr = pred.cpu().detach().numpy()
                            img_arr = pred_arr.astype("bool")
                            pred_polys = self.tile_ROIS(imgname=save_coords,mask_arr=img_arr)
                            polygons += pred_polys

                self.slide_ROIS(polygons=polygons,mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]))

            else:
                print("WARNING: YOU ENTERED AN INCORRECT MAGNIFICATION LEVEL")

        return

    def tile_ROIS(self,imgname,mask_arr):
        polygons = []
        nameparts = str.split(imgname, '_')
        pos = str.split(nameparts[0], '-')
        sz = str.split(nameparts[1], '-')
        radj = max([int(sz[0]), int(sz[1])]) / (self.save_image_size -1)
        start1 = int(pos[0])
        start2 = int(pos[1])
        c = morphology.remove_small_objects(mask_arr.astype(bool), 10, connectivity=2)
        c = morphology.binary_closing(c)
        c = morphology.remove_small_holes(c, 1000)
        contours, hier = cv2.findContours(c.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cvals = contour.transpose(0, 2, 1)
            cvals = np.reshape(cvals, (cvals.shape[0], 2))
            cvals = cvals.astype('float64')
            for i in range(len(cvals)):
                cvals[i][0] = start1 + radj * (cvals[i][0])
                cvals[i][1] = start2 + radj * (cvals[i][1])
            try:
                poly = Polygon(cvals)
                if poly.length > 0:
                    polygons.append(Polygon(poly.exterior))
            except:
                pass

        return polygons

    def slide_ROIS(self,polygons,mpp):
        all_polys = unary_union(polygons)
        final_polys = []
        for poly in all_polys:
            #print(poly)
            if poly.type == 'Polygon':
                newpoly = Polygon(poly.exterior)
                if newpoly.area*mpp*mpp > 12000:
                    final_polys.append(newpoly)
            if poly.type == 'MultiPolygon':
                for roii in poly.geoms:
                    newpoly = Polygon(roii.exterior)
                    if newpoly.area*mpp*mpp > 12000:
                        final_polys.append(newpoly)
        final_shape = unary_union(final_polys)

        trythis = '['
        for i in range(0, len(final_shape)):
            trythis += json.dumps(
                {"type": "Feature", "id": "PathAnnotationObject", "geometry": shapely.geometry.mapping(final_shape[i]),
                 "properties": {"classification": {"name": "Tumor", "colorRGB": -16711936}, "isLocked": False,
                                "measurements": []}}, indent=4)
            if i < len(final_shape) - 1:
                trythis += ','
        trythis += ']'

        with open(os.path.join(self.save_location,self.save_name+'_preds.json'), 'w') as outfile:
            outfile.write(trythis)

    def whitespace_check(self,im):
        bw = im.convert('L')
        bw = np.array(bw)
        bw = bw.astype('float')
        bw=bw/255
        prop_ws = (bw > 0.8).sum()/(bw>0).sum()
        return prop_ws




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_location')
    parser.add_argument('--image_file')
    parser.add_argument('--save_name')
    parser.add_argument('--save_dir')
    args = parser.parse_args()
    c = extractPatch()
    c.parseMeta_and_pullTiles()

