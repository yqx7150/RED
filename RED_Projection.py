import astra
import numpy as np

angle = 1 * np.pi
vol_geom = astra.create_vol_geom((360,360))
proj_geom = astra.create_proj_geom('parallel', 1.0, 360, np.linspace(0, angle,360,False))
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

def set_projection_size(vol_size = (360,360)):
    global vol_geom,proj_geom,proj_id,angle
    vol_geom = astra.create_vol_geom(vol_size)
    proj_geom = astra.create_proj_geom('parallel', 1.0, vol_size[0], np.linspace(0,angle,vol_size[1],False))
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

def pet_to_sino(image):

    sinogram_id, sinogram = astra.create_sino(image, proj_id)
    astra.data2d.delete(sinogram_id)

    return sinogram


def sino_to_pet(sinogram):
    global proj_id
    
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id
    cfg['option'] = {}
    cfg['option']['FilterType'] = 'hamming'

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    x = astra.data2d.get(rec_id)
    astra.data2d.delete([sinogram_id, rec_id])
    astra.algorithm.delete(alg_id)

    return x

