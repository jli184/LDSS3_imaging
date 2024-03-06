## photometry and astrometry calibration

## initial guess of wcs with astrometry.net
## source extractor -> scamp to align images with WCS
## swarp to align and stack images

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import subprocess

from astropy.wcs import WCS
from astropy.io import fits
from astropy.io.fits import Header
from astroquery.astrometry_net import AstrometryNet
from astropy.stats import sigma_clipped_stats
from astropy.constants import c,h

import seaborn as sns
import pandas as pd
from scipy.stats import norm
import time

from photutils.detection import DAOStarFinder
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier


def run_sourceextractor(image_name, weight_name, cat_name, seg_name, mag_zpt = 0., DETECT_THRESH=5.):

	'''
	Run source extractor.
	'''

	run = subprocess.run(['sex', image_name, '-c', 'source_ex.param', '-weight_image', weight_name, '-catalog_name', cat_name, '-checkimage_name', seg_name, '-mag_zeropoint', str(mag_zpt), '-DETECT_THRESH', str(DETECT_THRESH)])
	return;

def run_scamp(cat_name, ref_name, band, mosaic_type='LOOSE',mag_zpt = '0.', crossid_radius='5.0'):

	'''
	Run scamp. 
	'''

	run = subprocess.run(['scamp', cat_name, '-c', 'scamp_des.param', '-mosaic_type', mosaic_type, \
		'-ASTREFCAT_NAME', ref_name, '-MAGZERO_OUT', mag_zpt, '-CROSSID_RADIUS', crossid_radius])
	return;

def run_swarp(image_suffix, weight_suffix, coadd_image, coadd_weight, combine=True):

	'''
	Run swarp. 
	Options of (1) only resampling individual frames, and (2) resample and combine. 
	'''

	os.system('ls '+image_suffix+' >imagelist.txt')
	os.system('ls '+weight_suffix+' >weightlist.txt')
	if combine== True:	
		run = subprocess.run(['swarp', '@imagelist.txt', '-c', 'swarp_combine.param', '-weight_image', '@weightlist.txt', '-imageout_name', coadd_image, '-weightout_name', coadd_weight])
	else:
		run = subprocess.run(['swarp', '@imagelist.txt', '-c', 'swarp_single.param', '-weight_image', '@weightlist.txt'])

	run = subprocess.run(['rm','imagelist.txt'])
	run = subprocess.run(['rm','weightlist.txt'])

	return;

def make_DES_catalog(outname, ra, dec, radius, band):

	#query DES catalog in vizier and write to FITS_LDAC for scamp 

	## query
	vquery = Vizier(columns=['RA_ICRS', 'DE_ICRS','e_Aimg', 'e_Bimg', 'gmag', 'e_gmag', \
		'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 'ymag', 'e_ymag'], column_filters={"imag":"<23"}, row_limit=1e10, timeout=300)
	field = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg),frame='icrs')
	data = vquery.query_region(field, radius=radius*u.deg,catalog="II/371/des_dr2",cache=False)[0]
	data['e_ra_deg'] = (data['e_Aimg']*0.27)/60./60. ## convert pix uncertainty to deg (1pix = 0.27")
	data['e_dec_deg'] = (data['e_Bimg']*0.27)/60./60. ## convert pix uncertainty to deg (1pix = 0.27")
	data['mag'] = data[band+'mag']
	data['magerr'] = data['e_'+band+'mag']

	## create FITS_LDAC primary header (empty)
	primaryhdu = fits.PrimaryHDU(header=fits.Header())

	# create header table
	hdr_col = fits.Column(name='Field Header Card', format='1680A', array=["obtained through Vizier"])
	hdrhdu = fits.BinTableHDU.from_columns(fits.ColDefs([hdr_col]))
	hdrhdu.header['EXTNAME'] = ('LDAC_IMHEAD')
	hdrhdu.header['TDIM1'] = ('(80, 36)') # remove?

	# create data table
	colname_dic = {'RA_ICRS': 'XWIN_WORLD', 'DE_ICRS': 'YWIN_WORLD', 'e_ra_deg': 'ERRAWIN_WORLD',\
					'e_dec_deg': 'ERRBWIN_WORLD', 'mag': 'MAG', 'magerr': 'MAGERR'}
	format_dic = {'RA_ICRS': '1D', 'DE_ICRS': '1D', 'e_ra_deg': '1E', 'e_dec_deg': '1E', 'mag': '1E', 'magerr': '1E'}
	unit_dic = {'RA_ICRS': 'deg', 'DE_ICRS': 'deg', 'e_ra_deg': 'deg', 'e_dec_deg': 'deg', 'mag': 'mag', 'magerr': 'mag'}

	data_cols = []
	for col_name in data.columns:
		if col_name in list(colname_dic.keys()):
			data_cols.append(fits.Column(name=colname_dic[col_name], format=format_dic[col_name], \
			array=data[col_name], unit=unit_dic[col_name]))

	## add date; roughly DES reference epoch
	data_cols.append(fits.Column(name='OBSDATE', disp='F13.8', format='1D', unit='yr', array=np.ones(len(data))*2015.48))

	datahdu = fits.BinTableHDU.from_columns(fits.ColDefs(data_cols))
	datahdu.header['EXTNAME'] = ('LDAC_OBJECTS')

	# # combine HDUs and write file
	hdulist = fits.HDUList([primaryhdu, hdrhdu, datahdu])
	hdulist.writeto(outname, overwrite=True)

	return;

def make_DES_catalog_gold(outname, catname, band):

	#query DES catalog in vizier and write to FITS_LDAC for scamp 

	## query
	a = fits.open(catname)

	## create FITS_LDAC primary header (empty)
	primaryhdu = fits.PrimaryHDU(header=fits.Header())

	# create header table
	hdr_col = fits.Column(name='Field Header Card', format='1680A', array=["obtained through Vizier"])
	hdrhdu = fits.BinTableHDU.from_columns(fits.ColDefs([hdr_col]))
	hdrhdu.header['EXTNAME'] = ('LDAC_IMHEAD')
	hdrhdu.header['TDIM1'] = ('(80, 36)') # remove?

	data_cols = []
	data_cols.append(fits.Column(name='XWIN_WORLD', format='1D', array=a[1].data['ALPHAWIN_J2000'], unit='deg'))
	data_cols.append(fits.Column(name='YWIN_WORLD', format='1D', array=a[1].data['DELTAWIN_J2000'], unit='deg'))
	data_cols.append(fits.Column(name='ERRAWIN_WORLD', disp='F13.8', format='1E', unit='deg', array=np.ones(len(a[1].data['ALPHAWIN_J2000']))*1e-5))
	data_cols.append(fits.Column(name='ERRBWIN_WORLD', disp='F13.8', format='1E', unit='deg', array=np.ones(len(a[1].data['ALPHAWIN_J2000']))*1e-5))
	data_cols.append(fits.Column(name='MAG', format='1E', array=a[1].data['MAG_AUTO_'+band], unit='mag'))
	data_cols.append(fits.Column(name='MAGERR', format='1E', array=a[1].data['MAGERR_AUTO_'+band], unit='mag'))
	## add date; roughly DES reference epoch
	data_cols.append(fits.Column(name='OBSDATE', disp='F13.8', format='1D', unit='yr', array=np.ones(len(a[1].data['ALPHAWIN_J2000']))*2015.48))

	datahdu = fits.BinTableHDU.from_columns(fits.ColDefs(data_cols))
	datahdu.header['EXTNAME'] = ('LDAC_OBJECTS')

	# # combine HDUs and write file
	hdulist = fits.HDUList([primaryhdu, hdrhdu, datahdu])
	hdulist.writeto(outname, overwrite=True)

	return;


def add_wcs(image_name, weight_name, cat_name, seg_name, ref_name, band, cat_des, key='hqszpkingrxaqimp', mag_zpt='0.'):

	'''
	Add WCS to science images before combining with swarp. 
	First use Astrometry.net to 
	'''

	hdu = fits.open(image_name)
	image = hdu[0].data
	wei = fits.open(weight_name)
	weight = wei[0].data
	image_width = image.shape[0]
	image_height = image.shape[1]

	## if segmentation header does not exist, use source extractor & astrometry.net to find initial solution
	if os.path.exists(image_name[:-5]+'_wcs_init.fits') is False:
		
		print('Finding initial astrometry solution from astrometry.net...')
		#print('Running Source Extractor...')
		run_sourceextractor(image_name, weight_name, cat_name[:-4]+'.cat', seg_name[:-4]+'.seg', mag_zpt=mag_zpt)
		tmp = fits.open(cat_name[:-4]+'.cat')
		##note: astrometry.net seems much faster when the sources are sorted by brightness
		##		 too few stars (<20) will fail, but up to ~100 stars should be solved <20sec
		ind_sort = np.argsort(tmp[2].data['flux_auto'])[::-1]#[:100]
		print(len(ind_sort))
		
		## plot the image & the source detected
		#plt.figure()
		#norm = ImageNormalize(stretch=SqrtStretch())		
		#plt.imshow(image, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
		#positions = np.transpose((tmp[2].data['xwin_image'][ind_sort], tmp[2].data['ywin_image'][ind_sort]))
		#apertures = CircularAperture(positions, r=2.*np.median(tmp[2].data['kron_radius'][ind_sort]))
		#apertures.plot(color='red', lw=1.5, alpha=0.5)
		#positions = np.transpose((tmp[2].data['xwin_image'][ind][ind_sort], tmp[2].data['ywin_image'][ind][ind_sort]))
		#apertures = CircularAperture(positions, r=2.*np.median(tmp[2].data['kron_radius'][ind_sort]))
		#apertures.plot(color='blue', lw=1.5, alpha=0.5)
		#plt.show()
		#stop
		
		## find initial wcs from astrometry.net
		ast = AstrometryNet()
		ast.api_key = key	
		t1 = time.time()
		wcs_header = ast.solve_from_source_list(tmp[2].data['xwin_image'][ind_sort], tmp[2].data['ywin_image'][ind_sort], \
			image_width, image_height, solve_timeout=3000, publicly_visible='n')
		t2 = time.time()
		
		hdu_temp = hdu
		wcs = WCS(wcs_header).to_header()
		print(wcs)
		print('runtime for astrometry.net:',t2-t1)
		
		for x in wcs.cards: 
			hdu_temp[0].header.append(x)
		
		## if the astrometry.net solution failed, copy another frame's wcs as a initial guess
		if wcs['CRPIX1'] == 0 and wcs['CRPIX1'] == 0: 
			print('!!!Warning!!! Astrometry.net solution failed.')
			fname = glob.glob(image_name[:-15]+'*_wcs.fits')
			print(fname[0],fname)
			other_frame = fits.open(fname[0])

			for x in other_frame[0].header.cards:
				if x[0] in ['WCSAXES','CRPIX1','CRPIX2','CDELT1','CDELT2','CRVAL1','CRVAL2','CUNIT1','CUNIT2' \
				,'CTYPE1','CTYPE2','RADESYS','EQUINOX','CD1_1','CD1_2','CD2_1','CD2_2','PV1_1','PV1_2','PV1_4','PV1_5'\
				,'PV1_6','PV1_7','PV1_8','PV1_9','PV1_10','PV2_1','PV2_2','PV2_4','PV2_5','PV2_6','PV2_7','PV2_8'\
				,'PV2_9','PV2_10']: 
					try:
						hdu_temp[0].header[x[0]] = other_frame[0].header[x[0]]
					except KeyError: 
						hdu_tmp[0].header.append(x)
					except ValueError: 
						hdu_temp[0].header.append(x)

		hdu_temp.writeto(image_name[:-5]+'_wcs_init.fits',overwrite=True)

	## use scamp to calculate accurate wcs (can do multiple times to increase wcs accuracy)
	print('Running scamp to improve wcs accuracy...')
	n = 0
	diff_ra = 100.
	diff_dec = 100.
	os.system('cp '+image_name[:-5]+'_wcs_init.fits '+image_name[:-5]+'_wcs.fits')

	while (abs(diff_ra)>0.1 or abs(diff_dec)>0.1 or n<5):

		run_sourceextractor(image_name[:-5]+'_wcs.fits', weight_name, cat_name[:-4]+str(n)+'.cat', seg_name[:-4]+str(n)+'.seg')
		if os.path.exists(ref_name) is False:
			a = fits.open(image_name[:-5]+'_wcs_init.fits')
			make_DES_catalog_gold(ref_name, cat_des, band) ## search for 10' radius
		run_scamp(cat_name[:-4]+str(n)+'.cat',ref_name, band, mag_zpt=mag_zpt)

		orig = fits.open(image_name[:-5]+'_wcs.fits')
		try:
			ra_orig = orig[0].header['CRVAL1']-orig[0].header['CD1_1']*(image_width/2.-orig[0].header['CRPIX1'])#-orig[0].header['CD1_2']*(image_height/2.-orig[0].header['CRPIX2'])
			dec_orig = orig[0].header['CRVAL2']-orig[0].header['CD2_2']*(image_height/2.-orig[0].header['CRPIX2'])#-orig[0].header['CD2_1']*(image_width/2.-orig[0].header['CRPIX2'])
		except KeyError:
			ra_orig = orig[0].header['CRVAL1']-orig[0].header['PC1_1']*(image_width/2.-orig[0].header['CRPIX1'])#-orig[0].header['PC1_2']*(image_height/2.-orig[0].header['CRPIX2'])
			dec_orig = orig[0].header['CRVAL2']-orig[0].header['PC2_2']*(image_height/2.-orig[0].header['CRPIX2'])#-orig[0].header['PC1_2']*(image_height/2.-orig[0].header['CRPIX2'])		

		scamp_sol = open(cat_name[:-4]+str(n)+'.head','r')
		new_wcs = Header.fromstring(scamp_sol.read(),sep='\n')

		#print(new_wcs)

		for x in new_wcs.cards:
			if x[0]!='COMMENT': 
				try:
					orig[0].header[x[0]] = new_wcs[x[0]]
				except KeyError: 
					orig[0].header.append(x)
				except ValueError: 
					orig[0].header.append(x)		

		## clean up on additional WCS from astrometry.net, they interfere with scamp results
		for card in ['PC1_1','PC1_2','PC2_1','PC2_2','LATPOLE','LONPOLE','MJDREF']: 
			if card in orig[0].header: del orig[0].header[card]	

		ra_new = new_wcs['CRVAL1']-new_wcs['CD1_1']*(image_width/2.-new_wcs['CRPIX1'])#-new_wcs['CD1_2']*(image_height/2.-new_wcs['CRPIX2'])
		dec_new = new_wcs['CRVAL2']-new_wcs['CD2_2']*(image_height/2.-new_wcs['CRPIX2'])#-new_wcs['CD2_1']*(image_width/2.-new_wcs['CRPIX1'])
		diff_ra = (ra_orig-ra_new)*60.*60.#/diff_pix1
		diff_dec = (dec_orig-dec_new)*60.*60.#/diff_pix2
		n = n+1
		print('old:', ra_orig, dec_orig)
		print('new:', ra_new, dec_new)
		print('running sex+scamp ',str(n),' times:', diff_ra, diff_dec)

		orig.writeto(image_name[:-5]+'_wcs.fits',overwrite=True)
		#os.system('cp '+image_name[:-5]+'_wcs.fits '+image_name[:-5]+'_wcs_'+str(n)+'.fits')

		if n == 15: break

	
	return;


def check_astrometry(image, weight, catalog, write_catalog=True):

	## match to gaia to find stars
	run_swarp(image, weight, 'tmp.fits', 'tmp.weights.fits', combine=False)
	f = image.split('/')[-1]
	run_sourceextractor(f[:-5]+'.resamp.fits', f[:-5]+'.resamp.weight.fits', f[:-5]+'.cat', f[:-5]+'.seg')
	b = fits.open(f[:-5]+'.cat')
	ra_swarp = b[2].data['ALPHAWIN_J2000']
	dec_swarp = b[2].data['DELTAWIN_J2000']
	xx_swarp = b[2].data['XWIN_IMAGE']
	yy_swarp = b[2].data['YWIN_IMAGE']
	flag = b[2].data['flags']

	## match to GAIA stars to make sure we're checking FWHM with actual stars
	img = fits.open(f[:-5]+'.resamp.fits')
	coord = SkyCoord(ra=img[0].header['crval1'], dec=img[0].header['crval2'], unit=(u.degree, u.degree), frame='icrs')
	width = u.Quantity(0.3, u.deg)
	height = u.Quantity(0.3, u.deg)
	gstar = Gaia.query_object_async(coordinate=coord, width=width, height=height)

	#ra_gaia_match = np.array([])
	#dec_gaia_match = np.array([])
	ind_gaia_match = np.array([],dtype=int)
	for i in range(len(ra_swarp)):
		dist = np.sqrt((gstar['ra']-ra_swarp[i])**2+(gstar['dec']-dec_swarp[i])**2)*60.*60.
		ind = np.argmin(dist)
		if np.abs(ra_swarp[i]-gstar['ra'][ind])*3600.<0.5 and np.abs(dec_swarp[i]-gstar['dec'][ind])*3600.<0.5 and flag[i]==0:
			#ra_gaia_match = np.append(ra_gaia_match,ra_swarp[i])
			#dec_gaia_match = np.append(dec_gaia_match,dec_swarp[i])
			ind_gaia_match = np.append(ind_gaia_match,i)

	# now match to DES RA/DEC to align final astrometry
	a = fits.open(catalog)
	ra_des = a[1].data['ALPHAWIN_J2000']
	dec_des = a[1].data['DELTAWIN_J2000']

	res = np.array([])
	dra = np.array([])
	ddec = np.array([])
	ra_des_match = np.array([])
	dec_des_match = np.array([])
	xx_des_match = np.array([])
	yy_des_match = np.array([])
	ind_in_gaia = np.array([])

	for i in range(len(ra_swarp)):
		dist = np.sqrt((ra_swarp[i]-ra_des)**2+(dec_swarp[i]-dec_des)**2)*60.*60.
		#print(np.sort(dist)[:5])
		ind = np.argmin(dist)
		#if np.abs(ra_swarp[i]-ra_des[ind])*60.*60.<0.5 and np.abs(dec_swarp[i]-dec_des[ind])*60.*60.<0.5 and flag[i]==0:
		if np.min(dist)<0.5 and flag[i]==0:
			dra = np.append(dra, (ra_swarp[i]-ra_des[ind])*60.*60.)
			ddec = np.append(ddec, (dec_swarp[i]-dec_des[ind])*60.*60.)
			res = np.append(res, np.min(dist))
			ra_des_match = np.append(ra_des_match, ra_des[ind])
			dec_des_match = np.append(dec_des_match, dec_des[ind])
			xx_des_match = np.append(xx_des_match, xx_swarp[i])
			yy_des_match = np.append(yy_des_match, yy_swarp[i])

			if i in ind_gaia_match:
				ind_in_gaia = np.append(ind_in_gaia, 1)
			else:
				ind_in_gaia = np.append(ind_in_gaia, 0)

	## make hmsdms format for output
	ra_des_match_hms = np.array([],dtype=str)
	dec_des_match_dms = np.array([],dtype=str)
	for i in range(len(ra_des_match)):
		c = SkyCoord(ra_des_match[i]*u.deg, dec_des_match[i]*u.deg, frame='icrs')
		ra_des_match_hms = np.append(ra_des_match_hms, c.to_string('hmsdms').split(' ')[0].replace('h',':').replace('m',':')[:-1])
		dec_des_match_dms = np.append(dec_des_match_dms, c.to_string('hmsdms').split(' ')[1].replace('d',':').replace('m',':')[:-1])

	df = pd.DataFrame(data={'LDSS_x': xx_des_match, 'LDSS_y': yy_des_match, 'DES_RA_deg':ra_des_match, \
		'DES_Dec_deg':dec_des_match, 'DES_RA_hms':ra_des_match_hms, 'DES_RA_dms':dec_des_match_dms,\
		'dra':dra, 'ddec':ddec, 'd':res, 'in_gaia': ind_in_gaia})

	df_gaia = df.loc[df['in_gaia'] == 1]

	dra_mu, dra_med, dra_std = sigma_clipped_stats(df['dra'])
	ddec_mu, ddec_med, ddec_std = sigma_clipped_stats(df['ddec'])

	plt.rcdefaults()
	g = sns.jointplot(data=df, x='dra', y='ddec', kind="scatter",xlim=[-1.0,1.0],ylim=[-1.0,1.0],\
		marker='o',color='k',marginal_kws=dict(binrange=(-1,1),bins=20),joint_kws=dict(edgecolor='none',alpha=0.5))
	g.ax_joint.scatter('dra', 'ddec', data=df_gaia, c='r', marker='o',alpha=0.5)

	g.ax_joint.axvline(x=0,c='k')
	g.ax_joint.axhline(y=0,c='k')
	g.ax_joint.text(-0.9,0.85,'$\mu_x, \mu_y =$'+'{:.2f}'.format(dra_mu)+', '+'{:.2f}'.format(ddec_mu),fontsize=13)
	g.ax_joint.text(-0.9,0.75,'$\sigma_x, \sigma_y =$'+'{:.2f}'.format(dra_std)+', '+'{:.2f}'.format(ddec_std),fontsize=13)

	plt.savefig('/'.join(image.split('/')[:-1])+'/'+f[:-5]+'.astrometry1.pdf',bbox_inches='tight')

	plt.style.use('~/jli184.mplstyle')
	ind = np.where(df['in_gaia']==1)
	fig, ax = plt.subplots(2,2,figsize=(11,10))
	ax[0,0].plot(ra_des_match,dra,'ko',alpha=0.5,ms=5)
	ax[0,0].plot(ra_des_match[ind],dra[ind],'ro',ms=5)	
	ax[0,0].set_xlabel('RA')
	ax[0,0].set_ylabel('delta RA')
	ax[0,0].axhline(y=0,c='k')
	ax[0,1].plot(ra_des_match,ddec,'ko',alpha=0.5,ms=5)
	ax[0,1].plot(ra_des_match[ind],ddec[ind],'ro',ms=5)	
	ax[0,1].set_xlabel('RA')
	ax[0,1].set_ylabel('delta Dec')
	ax[0,1].axhline(y=0,c='k')
	ax[1,0].plot(dec_des_match,dra,'ko',alpha=0.5,ms=5)
	ax[1,0].plot(dec_des_match[ind],dra[ind],'ro',ms=5)	
	ax[1,0].set_xlabel('Dec')
	ax[1,0].set_ylabel('delta RA')
	ax[1,0].axhline(y=0,c='k')
	ax[1,1].plot(dec_des_match,ddec,'ko',alpha=0.5,ms=5)
	ax[1,1].plot(dec_des_match[ind],ddec[ind],'ro',ms=5)	
	ax[1,1].set_xlabel('Dec')
	ax[1,1].set_ylabel('delta Dec')
	ax[1,1].axhline(y=0,c='k')
	plt.tight_layout(w_pad=1.5)
	plt.savefig('/'.join(image.split('/')[:-1])+'/'+f[:-5]+'.astrometry2.pdf',bbox_inches='tight')

	#plt.savefig('/'.join(image_suffix.split('/')[:-1])+'/'+f[:-5]+'.pdf',bbox_inches='tight')
	if write_catalog == True:
		df.to_csv('/'.join(image.split('/')[:-1])+'/'+f[:-5]+'.des.csv',index=False,sep=' ')
		df_gaia.to_csv('/'.join(image.split('/')[:-1])+'/'+f[:-5]+'.gaia.csv',index=False,sep=' ')

	return;


def check_seeing(image1, weight1, image2, weight2):

	plt.style.use('~/jli184.mplstyle')

	'''
	Check astrometry in individual science frames.
	Use swarp to make resampled images and check with a catalog. 
	Output is saved as a pdf file with d_ra, d_dec 1d and 2d histograms. 
	'''
	
	## first image
	run_swarp(image1, weight1, None, None, combine=False)
	run_sourceextractor(image1.split('/')[-1][:-5]+'.resamp.fits', image1.split('/')[-1][:-5]+'.resamp.weight.fits', 'tmp1.cat', 'tmp1.seg')
	a = fits.open('tmp1.cat')
	ra1 = a[2].data['ALPHAWIN_J2000']
	dec1 = a[2].data['DELTAWIN_J2000']
	fwhm1 = a[2].data['FWHM_IMAGE']
	flag1 = a[2].data['flags']
	isstar1 = a[2].data['class_star']

	## match to GAIA stars to make sure we're checking FWHM with actual stars
	img = fits.open(image1.split('/')[-1][:-5]+'.resamp.fits')
	coord = SkyCoord(ra=img[0].header['crval1'], dec=img[0].header['crval2'], unit=(u.degree, u.degree), frame='icrs')
	width = u.Quantity(0.3, u.deg)
	height = u.Quantity(0.3, u.deg)
	gstar = Gaia.query_object_async(coordinate=coord, width=width, height=height)

	ra_match1 = np.array([])
	dec_match1 = np.array([])
	fwhm_match1 = np.array([])
	g1 = np.array([])
	for i in range(len(ra1)):
		dist = np.sqrt((gstar['ra']-ra1[i])**2+(gstar['dec']-dec1[i])**2)*60.*60.
		ind = np.argmin(dist)
		if np.abs(ra1[i]-gstar['ra'][ind])*3600.<0.2 and np.abs(dec1[i]-gstar['dec'][ind])*3600.<0.2 and flag1[i]==0 and gstar['phot_g_mean_mag'][ind]>18:
			#print(ind,ra1[i],gstar['ra'][ind],dec1[i],gstar['dec'][ind],gstar['phot_g_mean_mag'][ind],fwhm1[i],isstar1[i])
			fwhm_match1 = np.append(fwhm_match1,fwhm1[i])
			ra_match1 = np.append(ra_match1,ra1[i])
			dec_match1 = np.append(dec_match1,dec1[i])
			g1 = np.append(g1,gstar['phot_g_mean_mag'][ind])


	print(len(gstar),len(fwhm_match1))

	## second image
	run_swarp(image2, weight2, None, None, combine=False)
	run_sourceextractor(image2.split('/')[-1][:-5]+'.resamp.fits', image2.split('/')[-1][:-5]+'.resamp.weight.fits', 'tmp2.cat', 'tmp2.seg')
	b = fits.open('tmp2.cat')
	ra2 = b[2].data['ALPHAWIN_J2000']
	dec2 = b[2].data['DELTAWIN_J2000']
	fwhm2 = b[2].data['FWHM_IMAGE']
	flag2 = b[2].data['flags']
	isstar2 = b[2].data['class_star']

	## match to GAIA stars to make sure we're checking FWHM with actual stars
	ra_match2 = np.array([])
	dec_match2 = np.array([])
	fwhm_match2 = np.array([])
	g2 = np.array([])
	for i in range(len(ra2)):
		dist = np.sqrt((gstar['ra']-ra2[i])**2+(gstar['dec']-dec2[i])**2)*60.*60.
		ind = np.argmin(dist)
		if np.abs(ra2[i]-gstar['ra'][ind])*3600.<0.2 and np.abs(dec2[i]-gstar['dec'][ind])*3600.<0.2 and flag2[i]==0 and gstar['phot_g_mean_mag'][ind]>18:
			#print(ind,ra2[i],gstar['ra'][ind],dec2[i],gstar['dec'][ind],gstar['phot_g_mean_mag'][ind],fwhm2[i],isstar2[i])
			fwhm_match2 = np.append(fwhm_match2,fwhm2[i])
			ra_match2 = np.append(ra_match2,ra2[i])
			dec_match2 = np.append(dec_match2,dec2[i])
			g2 = np.append(g2,gstar['phot_g_mean_mag'][ind])

	print(len(gstar),len(fwhm_match2))

	fwhm_match1 = fwhm_match1*0.188
	fwhm_match2 = fwhm_match2*0.188


	plt.figure(figsize=(6,6))
	n1, bins, patches  = plt.hist(fwhm_match1,bins=30,range=(0,3),color='r',histtype='step',label='Single')
	plt.axvline(x=np.median(fwhm_match1),c='r',lw=2,ls='--')
	n2, bins, patches  = plt.hist(fwhm_match2,bins=30,range=(0,3),color='k',histtype='step',label='Combined')
	plt.axvline(x=np.median(fwhm_match2),c='k',lw=2,ls='--')
	#plt.legend(loc='upper right')
	plt.text(2,np.max([n1,n2])*1.0,'Single: '+'{:.2f}'.format(np.median(fwhm_match1)),color='r',fontsize=14)
	plt.text(2,np.max([n1,n2])*1.1,'Coadd: '+'{:.2f}'.format(np.median(fwhm_match2)),color='k',fontsize=14)

	plt.xlim([0,3])
	plt.ylim([0,np.max([n1,n2])*1.2])
	plt.xlabel('FWHM (arcsec)')
	plt.ylabel('N')
	plt.savefig('/'.join(image1.split('/')[:-1])+'/'+image1.split('/')[-1][:-5]+'.seeing.pdf',bbox_inches='tight')

	return;

'''
def check_depth(image, weight, zpt):

	
	Calculate the 10-sigma limit and source extractor turn-over point.
	Can be used to estimate image depth & compare with other images/catalogs.
	

	## rms in image
	a = fits.open(image)
	img = a[0].data
	bkg_mu, bkg_med, bkg_std = sigma_clipped_stats(img.flatten()[np.where(img.flatten()!=0)])

	lim = ((1.0/0.188)**2)*(bkg_med+bkg_std)*a[0].header['gain']/a[0].header['exptime']
	#lim_flux = lim*h.value*c.value/(wave_eff*1e-10)*1e7 ## E = hc/lambda
	#lim_mag = -2.5*np.log10(lim_flux)-48.6 ## flux to mag
	lim_mag = -2.5*np.log10(lim)+zpt

	#print(bkg_mu, bkg_med, bkg_std)
	if os.path.exists(image[:-5]+'.cat') is False:
		run_sourceextractor(image, weight, image[:-5]+'.cat', image[:-5]+'.seg', mag_zpt = zpt)

	b = fits.open(image[:-5]+'.cat')
	ind = np.where(b[2].data['flags']==0)
	lim_sex = -2.5*np.log10(b[2].data['flux_auto'][ind]*a[0].header['gain']/a[0].header['exptime'])+zpt

	print('Image depth (mag/arcsec2):', lim_mag)
	print('Source Extractor limit (10-sigma):', np.max(lim_sex))
	#plt.plot(-2.5*np.log10(b[2].data['flux_auto']*a[0].header['gain']/a[0].header['exptime'])+zpt,b[2].data['class_star'],'k.')
	#plt.plot(-2.5*np.log10(b[2].data['flux_auto'][ind]*a[0].header['gain']/a[0].header['exptime'])+zpt,b[2].data['class_star'][ind],'r.')
	#plt.show()

	return;
'''

def check_depth_DES(image, weight, catalog, filtername, zpt):

	plt.style.use('~/jli184.mplstyle')

	#if os.path.exists(image[:-5]+'.cat') is False:
	run_sourceextractor(image, weight, image[:-5]+'.cat', image[:-5]+'.seg', mag_zpt = 0., DETECT_THRESH=1.5)

	a = fits.open(image)
	b = fits.open(image[:-5]+'.cat')
	ra_image = b[2].data['ALPHAWIN_J2000']
	dec_image = b[2].data['DELTAWIN_J2000']
	xx_image = b[2].data['XWIN_IMAGE']
	yy_image = b[2].data['YWIN_IMAGE']
	flag = b[2].data['flags']
	mag_image = -2.5*np.log10(b[2].data['flux_auto']*a[0].header['gain']/a[0].header['exptime'])+zpt
	print(np.max(mag_image[np.isfinite(mag_image)]))

	## match to DES catalog
	c = fits.open(catalog)
	ra_des = c[1].data['ALPHAWIN_J2000']
	dec_des = c[1].data['DELTAWIN_J2000']
	mag_des = c[1].data['mag_auto_'+filtername]
	
	mag_des_match = np.array([])
	mag_image_match = np.array([])

	for i in range(len(ra_image)):
		dist = np.sqrt((ra_image[i]-ra_des)**2+(dec_image[i]-dec_des)**2)*60.*60.
		#print(np.sort(dist)[:5])
		ind = np.argmin(dist)
		if np.abs(ra_image[i]-ra_des[ind])*60.*60.<0.5 and np.abs(dec_image[i]-dec_des[ind])*60.*60.<0.5 and flag[i]==0:
			mag_des_match = np.append(mag_des_match, mag_des[ind])
			mag_image_match = np.append(mag_image_match, mag_image[i])

	dmag_mu, dmag_med, dmag_std = sigma_clipped_stats(mag_des_match-mag_image_match)
	zpt_des = zpt+dmag_med

	plt.figure(figsize=(6,6))
	plt.hist(mag_image+dmag_med,histtype='step',bins=np.arange(16,26.01,0.2),label='LDSS',color='k')
	print(np.percentile(mag_image[np.isfinite(mag_image)]+dmag_med,[16,50,84,99]))
	#plt.legend()
	plt.xlabel('Magnitude',fontsize=16)
	plt.ylabel('N',fontsize=16)
	plt.savefig('/'.join(image.split('/')[:-1])+'/'+image.split('/')[-1][:-5]+'.mag.pdf',bbox_inches='tight')

	img = fits.open(image)
	img[0].header['ZPT_DES'] = (np.round(zpt_des,2), 'zeropoint from matching to DES')
	img.writeto(image, overwrite=True)

	bkg_mu, bkg_med, bkg_std = sigma_clipped_stats(img[0].data.flatten()[np.where(img[0].data.flatten()!=0)])
	lim = ((1.0/img[0].header['scale'])**2)*(bkg_med+bkg_std)*a[0].header['gain']/a[0].header['exptime']
	lim_mag = -2.5*np.log10(lim)+zpt

	print('zeropoint from DES', zpt_des)
	print('10 sigma limit', np.round(lim_mag,1), 'mag/arcsec2')

	return;


def clean_tmp(dir_cal, targetname, filtername):
	
	'''
	Remove temporary files. 
	This includes: reduced single-frame CCD science images, outputs from source extractor and scamp.
	'''

	os.system('rm -r '+dir_cal+'*'+targetname+'*'+filtername+'*c1.fits')
	os.system('rm -r '+dir_cal+'*'+targetname+'*'+filtername+'*c2.fits')
	os.system('rm -r '+dir_cal+'*'+targetname+'*'+filtername+'*.head')
	os.system('rm -r '+dir_cal+'*'+targetname+'*'+filtername+'*.seg')
	os.system('rm -r '+dir_cal+'*'+targetname+'*'+filtername+'*.cat')

	return;

