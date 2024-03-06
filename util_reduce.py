## LDSS imaging pipeline
## procedure:
## 		- make a list of filenames and file type
## 		- reduce the data: trim and reduced the bias, flat, and make master bias & flat images
##		- reduce science frame
## 		- stitch the science and flat frames together

import matplotlib.pyplot as plt
import numpy as np
import os
import glob

from astropy.modeling import models
from astropy import units as u
from astropy import nddata
from astropy.io import fits

import ccdproc
from ccdproc import ImageFileCollection
import re
from astropy.stats import mad_std
from astropy.table import Table


def inv_median(a):
    return 1 / np.median(a)

def read_exposure(dir_raw, outname):

	'''
	Make a list of exposures in 'dir_raw', and print summary in the file 'outname'
	If 'outname' exists, read directly from the file.

	After 'outname' is generated for the first time, bad exposures can be flagged manually in the file.

	Note: glob_include ccd*c* to include only individual chips (not combined ones)
	'''

	keys = ['date-obs', 'time-obs', 'instrume', 'airmass', 'exptime', 'object', 'exptype', 'filter', 'grism', 'opamp']
	if os.path.exists(outname) is False: 
		explist = ImageFileCollection(dir_raw, keywords=keys, glob_include='ccd*c*')
		explist.summary.write(outname,format='ascii')
	else:
		dat = Table.read(outname, format='ascii')
		explist = ImageFileCollection(dir_raw, keywords=keys, filenames=list(dat['file']))

	return explist;

def oscan_and_trim(image_list):

	'''
	Trim the overscan region (overscan is not useful in LDSS data, so there is no overscan subtraction).
	The original image list is replaced by a list of images with the changes applied.
	'''

	for idx, img in enumerate(image_list):
		image_list[idx] = ccdproc.trim_image(img, fits_section=img.header['datasec'], add_keyword={'trimmed': True, 'calstat': 'T'})
	return image_list;

def make_bias(explist, dir_cal, plot_fig=True):

	'''
	Reduce the bias frames. The bias frames are trimmed and then median-combined.
	If plot_fig is True, the bias image will be saved in a pdf file. 
	'''

	print('Making master bias images...')
	chips = set(h['opamp'] for h in explist.headers(exptype='bias'))

	for c in chips:	
		print('	combining bias for chip '+str(c)+' / '+str(len(chips)))
		bias_list = []
		for hdu, fname in explist.hdus(exptype='bias',opamp=c,return_fname=True):
			meta = hdu.header
			meta['filename'] = fname
			bias_list.append(ccdproc.CCDData(data=hdu.data, meta=meta, unit="adu"))

		print('	# of bias frames:', len(bias_list))
		bias_list = oscan_and_trim(bias_list)

		biases = ccdproc.Combiner(bias_list)
		master_bias = biases.median_combine()

		print('	bias stats (16/50/84%):', np.percentile(master_bias,[16,50,84]))

		if plot_fig == True:
			plt.figure()
			plt.imshow(master_bias, vmax=np.percentile(master_bias,84), vmin=np.percentile(master_bias,16),origin='lower')
			plt.colorbar()
			plt.savefig(dir_cal+'master_bias_c'+str(c)+'.pdf',bbox_inches='tight')
			#plt.show()

		master_bias.meta['combined'] = True
		master_bias.write(dir_cal+'master_bias_c'+str(c)+'.fits')

	return;

def make_flat(explist, dir_cal, flattype, plot_fig=True, skyflat=False, exptime_lim=[100,600], **flat_filters):

	'''
	Reduce the flat frames. The flat frames are trimmed and then combined.
	The images are weighted by the inv_median scale, with 5 sigma clipping, and median-combined.
	If plot_fig is True, the bias image will be saved in a pdf file. 
	'''

	## reading in the combined bias
	explist_reduced = ccdproc.ImageFileCollection(dir_cal,glob_include='master_bias*')
	bias_list = {}
	for hdu, fname in explist_reduced.hdus(return_fname=True):
		meta = hdu.header
		meta['filename'] = fname
		bias_list[fname] = ccdproc.CCDData(data=hdu.data, meta=meta, unit="adu")

	print('Making master flat images...')
	chips = set(h['opamp'] for h in explist.headers(**flat_filters))
	filters = set(h['filter'] for h in explist.headers(**flat_filters))

	if skyflat==True:
		exptime = np.unique([h['exptime'] for h in explist.headers(**flat_filters)])
		exptime = exptime[np.where(np.all([exptime>exptime_lim[0],exptime<exptime_lim[1]],axis=0))[0]]
		print(exptime)

	for f in filters:
		for c in chips:	
			print('	combining flat for chip '+str(c)+' / '+str(len(chips))+', filter: '+str(f))
			flat_list = []
			if skyflat == True:
				for t in exptime:
					flat_filters['exptime'] = t
					for hdu, fname in explist.hdus(opamp=c,filter=f,return_fname=True,**flat_filters):
						print(fname)
						meta = hdu.header
						meta['filename'] = fname
						flat_list.append(ccdproc.CCDData(data=hdu.data, meta=meta, unit="adu"))
			else:
				for hdu, fname in explist.hdus(opamp=c,filter=f,return_fname=True,**flat_filters):
					print(fname)
					meta = hdu.header
					meta['filename'] = fname
					flat_list.append(ccdproc.CCDData(data=hdu.data, meta=meta, unit="adu"))

			print('	# of flats: ',len(flat_list))

			if len(flat_list)>0:
				## overscan and trimming
				flat_list = oscan_and_trim(flat_list)
				for idx, img in enumerate(flat_list):
					flat_list[idx] = ccdproc.subtract_bias(img, bias_list['master_bias_c'+str(c)+'.fits'])

				## combine flats
				master_flat = ccdproc.combine(flat_list,method='median', scale=inv_median, sigma_clip=True, \
					sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,sigma_clip_func=np.ma.median, \
					signma_clip_dev_func=mad_std)

				med = np.median(master_flat.data[np.where(master_flat.data>0.5)])
				print('	flat stats (16/50/84%):', np.percentile(master_flat,[16,50,84]), med)
				master_flat.data[np.where(master_flat.data>1.1*med)] = 0.
				master_flat.data[np.where(master_flat.data<0.9*med)] = 0.

				print('	flat stats (16/50/84%):', np.percentile(master_flat,[16,50,84]))

				master_flat.meta['combined'] = True
				master_flat.write(dir_cal+'master_'+flattype+'flat_'+f+'_c'+str(c)+'.fits',overwrite=True)

				if plot_fig == True:
					plt.figure()
					plt.imshow(master_flat, vmax=1.1*np.median(med), vmin=0.9*np.median(med))
					plt.colorbar()
					plt.savefig(dir_cal+'master_'+flattype+'flat_'+f+'_c'+str(c)+'.pdf',bbox_inches='tight')


	return;

def reduce_science(explist, targetname, filtername, dir_cal, flattype='sky'):

	'''
	Reduce the individual science frames, and make weight images for each frame. 
	The science frames are trimmed (not combined).
	The bad pixels are assigned 0 in the weight images.
	'''

	## reading in the combined bias
	explist_reduced = ccdproc.ImageFileCollection(dir_cal,glob_include='master_bias*')
	bias_list = {}
	for hdu, fname in explist_reduced.hdus(return_fname=True):
		meta = hdu.header
		meta['filename'] = fname
		bias_list[fname] = ccdproc.CCDData(data=hdu.data, meta=meta, unit="adu")

	## reading in the combined flats
	explist_reduced = ccdproc.ImageFileCollection(dir_cal,glob_include='master_'+flattype+'flat*')
	flat_list = {}
	for hdu, fname in explist_reduced.hdus(return_fname=True):
		meta = hdu.header
		meta['filename'] = fname
		flat_list[fname] = ccdproc.CCDData(data=hdu.data, meta=meta, unit="adu")


	## start making science images
	print('Making science image ...')
	explist.refresh()
	chips = set(h['opamp'] for h in explist.headers({'exptype': 'object', 'filter': filtername}))
	#print(chips)

	## TODO: make name selection more flexible? 
	name_all = np.unique([h['object'] for h in explist.headers({'exptype': 'object', 'filter': filtername})])
	print(name_all)
	name_tmp = []
	for x in name_all:
		if x==targetname+' i image': name_tmp.append(x)
	#print(name_tmp)

	for c in chips:	
		print('	reducing for chip '+str(c)+' / '+str(len(chips))+', filter: '+str(filtername)+', target:'+str(targetname))
		sci_list = []
		weight_list = []
		for t in name_tmp:
			for hdu, fname in explist.hdus(opamp=c,filter=filtername,object=t,return_fname=True):
				#print(fname)
				meta = hdu.header
				meta['filename'] = fname
				sci_list.append(ccdproc.CCDData(data=hdu.data, meta=meta, unit="adu"))

				weight_list.append(ccdproc.CCDData(data=np.zeros(hdu.data.shape), meta=meta, unit="adu"))

		print('	# of images: ',len(sci_list))

		
		## overscan and trimming
		if len(sci_list)>0:
			sci_list = oscan_and_trim(sci_list)
			for idx, img in enumerate(sci_list):
				img = ccdproc.subtract_bias(img, bias_list['master_bias_c'+str(c)+'.fits'], add_keyword={'biassub': True, 'calstat': 'TB'})
				sci_list[idx] = ccdproc.flat_correct(img, flat_list['master_'+flattype+'flat_'+filtername+'_c'+str(c)+'.fits'], \
					add_keyword={'flatcorr': True, 'calstat': 'TBF'})
				sci_list[idx].data[np.where(flat_list['master_'+flattype+'flat_'+filtername+'_c'+str(c)+'.fits'].data<0.1)] = 0.
				sci_list[idx].write(dir_cal+'reduced_'+targetname.split(' ')[0]+'_'+filtername+'_'+str(idx)+'_c'+str(c)+'.fits',overwrite=True)

				## make weight map
				weight_list[idx].data = 1./(sci_list[idx].data*meta['egain'])
				weight_list[idx].data[np.where(flat_list['master_'+flattype+'flat_'+filtername+'_c'+str(c)+'.fits'].data<0.1)] = 0.
				weight_list[idx].write(dir_cal+'weight_'+targetname.split(' ')[0]+'_'+filtername+'_'+str(idx)+'_c'+str(c)+'.fits',overwrite=True)
		
	return;


def stitch_frame(prefix, imagesize=(4096,2048), chip=[1,2], normalize=True, remove_old=True):

	'''
	Stitch the individual CCD images. 
	## TODO: need flexibility to stitch IMACS (8 CCDs) too. 
	'''

	new_image = np.zeros(imagesize).T

	a = fits.open(prefix+'_c1.fits')
	b = fits.open(prefix+'_c2.fits')

	## manually adjust the background level due to slight difference in CCD
	sky_a = np.median(a[0].data[np.where(np.all([np.isfinite(a[0].data),a[0].data!=0],axis=0))])
	sky_b = np.median(b[0].data[np.where(np.all([np.isfinite(b[0].data),b[0].data!=0],axis=0))])
	#print(sky_a, sky_b)
				
	ny, nx = a[0].data.shape
	#print (nx, ny)
	## rescale to match the backgrounds in two frames
	new_image[:,nx:] = a[0].data[:,::-1]
	if normalize == True:
		new_image[:,:nx] = b[0].data*sky_a/sky_b
	else:
		new_image[:,:nx] = b[0].data


	#print(new_image.shape)
	head = a[0].header
	head['opamp']='combined'
	head['DATASEC'] = '[1:2048,1:4096]' 
	head['BIASSEC'] = 'none'
	hdu = fits.PrimaryHDU(data=np.array(new_image,dtype='float'),header=head)
	hdu.writeto(prefix+'_combined.fits',overwrite=True)

	if remove_old == True:
		os.system('rm '+prefix+'_c1.fits')
		os.system('rm '+prefix+'_c2.fits')

	return;


