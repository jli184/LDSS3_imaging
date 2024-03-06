## execution of the LDSS pipeline

import util_reduce as util_reduce
import util_align_single as util_align
import sys
import os 
import numpy as np
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u


##########################
## Running the pipeline ##
##########################

## python run_pipeline.py dir_raw arg1 arg2 arg3
## arg1:
##		init: make exposure list
##		reduce: reduce all calibration & science frames 
##		image: combine all science images 
## arg2: 
##		for "reduce": flattype
##		for "image": object name
## arg3:
##		for "reduce": obj (target name for twi/ambient/dome flat, don't need for sky flat)
##		for "image": filter name

print(sys.argv)

dir_raw = sys.argv[1]
dir_cal = dir_raw[:-1]+'_reduced/'
if len(sys.argv)>2: 
	action = sys.argv[2]
else:
	action = 'init'

#
# make directory for data reduction and exposure list
#
os.system('mkdir '+dir_cal)
explist = util_reduce.read_exposure(dir_raw, dir_cal+'exposure.txt')
#print(explist.summary)

#
# reduce calibration and science frames
#
if action == 'reduce':

	## make bias
	master_bias = util_reduce.make_bias(explist, dir_cal)
	flattype = sys.argv[3]

	## make flats
	if flattype == 'twi':
		flatname = sys.argv[4]
		flat_filters = {'exptype': 'flat', 'object': flatname}
		master_flat = util_reduce.make_flat(explist, dir_cal, flattype, **flat_filters)

	if flattype == 'sky':
		flat_filters = {'exptype': 'object'}
		master_flat = util_reduce.make_flat(explist, dir_cal, flattype, skyflat=True, **flat_filters)

	if flattype == 'ambient' or flattype == 'dome':
		flatname = sys.argv[4]
		flat_filters = {'exptype': 'object', 'object': flatname}
		master_flat = util_reduce.make_flat(explist, dir_cal, flattype, **flat_filters)
		

#
# combine science images
#
if action == 'image':

	## calibrate science frames
	print('WARNING: check target name in log, might have to edit code to select correct exposure files.')
	if sys.argv[3] == 'all':
		target_tmp = np.unique([h['object'] for h in explist.headers({'grism':'open','exptype':'object'})])
		print(target_tmp)
		targets = []
		for t in target_tmp: 
			if t[0]=='J' : targets.append(t[:5])
		targets = np.unique(targets)
		print('All targets:', targets)
	else:
		targets = [sys.argv[3]]
		print('Single target:', targets)

	
	if sys.argv[4] == 'all':
		filters = list(set([h['filter'] for h in explist.headers({'grism':'open','exptype':'object'})]))
		filters.remove('Open')
		#filters.remove('r_Sloan') ## remove other filters manually if needed
		#filters.remove('GG495')	## remove other filters manually if needed
		print('All filters:', filters)
	else:
		filters = [sys.argv[4]]
		print('Single filter:', filters)
	
	
	zpt = {'z_Sloan':27.81, 'i_Sloan':27.81, 'r_Sloan':27.82, 'g_Sloan':27.65} ## from LCO website

	for tt in targets:
		for ff in filters:
			
			print('running science frames:', tt, ff)
			
			
			sci_image = util_reduce.reduce_science(explist, tt, ff, dir_cal) 

			## stitch science frames
			
			fname = glob.glob(dir_cal+'reduced_'+tt+'*'+ff+'*')
			fname = np.unique([f[:-8] for f in fname])
			print (fname)
			for f in fname: 
				#print(f)
				util_reduce.stitch_frame(f, imagesize=(2048,4096), chip=[1,2])

			fname = glob.glob(dir_cal+'weight_'+tt+'*'+ff+'*')
			fname = np.unique([f[:-8] for f in fname])
			#print (fname)
			for f in fname: 
				#print(f)
				util_reduce.stitch_frame(f, imagesize=(2048,4096), chip=[1,2])
			

			##
			## add wcs
			##

			if tt[0]!='J':
				cat_des = glob.glob('/Users/jennili/Research/CUBS/CUBS_MUSE_reduced/catalog/DES_CUBS/J'+tt+'*-v4.fits')[0]
			else:
				cat_des = glob.glob('/Users/jennili/Research/CUBS/CUBS_MUSE_reduced/catalog/DES_CUBS/'+tt+'*-v4.fits')[0]

			fname = glob.glob(dir_cal+'reduced_'+tt+'_'+ff+'*_combined.fits')
			#print(fname)
			for f in fname: 
				#print(f)
				if os.path.exists(f[:-5]+'_wcs.fits') is False:
					weight_name = f.replace('reduced_','weight_')
					cat_name = f[:-5]+'.cat'
					seg_name = f[:-5]+'.seg'
					hdu = fits.open(f)
					ref_name = dir_cal+'DES_Y3gold_'+tt+'_'+ff+'.fits'
					util_align.add_wcs(f, weight_name, cat_name, seg_name, ref_name, ff[0], cat_des, mag_zpt=str(zpt[ff]))
					util_align.check_astrometry(f[:-5]+'_wcs.fits', weight_name, cat_des, write_catalog=False)	

			##
			## coadd image
			##
			
			image_suffix = dir_cal+'reduced_'+tt+'_'+ff+'*combined_wcs.fits'
			weight_suffix = dir_cal+'weight_'+tt+'_'+ff+'*combined.fits'
			coadd_image = dir_cal+tt+'_'+ff+'_image.fits'
			coadd_weight = dir_cal+tt+'_'+ff+'_weight.fits'
			if os.path.exists(coadd_image) is False: 
				util_align.run_swarp(image_suffix, weight_suffix, coadd_image, coadd_weight)

				## add individual exposure time to the final image header
				coadd = fits.open(coadd_image)
				coadd[0].header['TOT_TIME'] = (coadd[0].header['EXPTIME'], coadd[0].header.comments['EXPTIME'])
				tmp_exptime = []
				for i in range(50):
					if 'FILE'+str(i).zfill(4) in coadd[0].header:
						ind_image = fits.open(dir_cal+coadd[0].header['FILE'+str(i).zfill(4)])
						coadd[0].header['EXPT'+str(i).zfill(4)] = ind_image[0].header['exptime']
						tmp_exptime.append(ind_image[0].header['exptime'])
				coadd[0].header['EXPTIME'] = (np.median(tmp_exptime), 'Median individual exposure time (s)')
				coadd.writeto(coadd_image,overwrite=True)
				util_align.check_astrometry(coadd_image, coadd_weight, cat_des)
			
			
			##
			## compare seeing in final images
			##
			coadd_image = dir_cal+tt+'_'+ff+'_image.fits'
			coadd_weight = dir_cal+tt+'_'+ff+'_weight.fits'
			fnames = glob.glob(dir_cal+'reduced_'+tt+'_'+ff+'*_combined_wcs.fits')
			for f in fnames:
				util_align.check_seeing(f, f.replace('reduced_','weight_').replace('_wcs',''), coadd_image, coadd_weight)
			util_align.check_depth_DES(coadd_image,coadd_weight,cat_des,ff[0],zpt[ff])
						
			util_align.clean_tmp(dir_cal, tt, ff)
			os.system('rm -r *.fits')
			os.system('rm -r tmp*')
			os.system('rm -r *.cat')
			os.system('rm -r *.seg')			

			
			