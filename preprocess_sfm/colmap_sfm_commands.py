import os
import subprocess
import multiprocessing

from preprocess_sfm.colmap_sfm_utils import init_posed_sfm

gpu_index = '-1'

COLMAP_BIN = os.path.join(os.path.dirname(__file__), 'ColmapForVisSat/build/__install__/bin/colmap')
COLMAP_ENV = os.environ.copy()
COLMAP_ENV['LD_LIBRARY_PATH'] = os.path.join(os.path.dirname(__file__), 'ColmapForVisSat/build/__install__/lib')

assert (os.path.isfile(COLMAP_BIN)), 'Please install ColmapForVisSat first; '\
        'check {} for instructions'.format(os.path.join(os.path.dirname(__file__), 'install_colmapforvissat.sh '))


def bash_run(cmd):
    subprocess.check_call(['/bin/bash', '-c', cmd], env=COLMAP_ENV)


def run_sift_matching(img_dir, db_file, debug_folder=None):
    '''note: on cluster without display, feature extraction and matching can only run in cpu mode'''
    if os.path.exists(db_file): # otherwise colmap will skip sift matching
        os.remove(db_file)
    # feature extraction
    cmd = COLMAP_BIN + ' feature_extractor --database_path {} \
                                    --image_path {} \
                                    --ImageReader.camera_model PERSPECTIVE \
                                    --SiftExtraction.max_image_size 10000  \
                                    --SiftExtraction.estimate_affine_shape 0 \
                                    --SiftExtraction.domain_size_pooling 1 \
                                    --SiftExtraction.max_num_features 20000 \
                                    --SiftExtraction.num_threads {} \
                                    --SiftExtraction.use_gpu 1 \
                                    --SiftExtraction.gpu_index {}'.format(db_file, img_dir, multiprocessing.cpu_count(), gpu_index)
    if debug_folder is not None:
        cmd += '  2>&1 | tee {}'.format(os.path.join(os.path.join(debug_folder, 'log_feature_extraction.txt')))
    bash_run(cmd)
    # feature matching
    cmd = COLMAP_BIN + ' exhaustive_matcher --database_path {} \
                                        --SiftMatching.guided_matching 1 \
                                        --SiftMatching.num_threads 6 \
                                        --SiftMatching.max_error 3 \
                                        --SiftMatching.max_num_matches 50000 \
                                        --SiftMatching.use_gpu 1 \
                                        --SiftMatching.gpu_index {}'.format(db_file, gpu_index)
    if debug_folder is not None:
        cmd += '  2>&1 | tee {}'.format(os.path.join(os.path.join(debug_folder, 'log_feature_matching.txt')))
    bash_run(cmd)


def run_point_triangulation(img_dir, db_file, out_dir, cam_dict_file,
                            tri_merge_max_reproj_error, tri_complete_max_reproj_error, filter_max_reproj_error,
                            debug_folder=None):
    os.makedirs(out_dir, exist_ok=True)

    # create initial poses
    init_posed_sfm(db_file, cam_dict_file, out_dir)
    
    # triangulate points
    cmd = COLMAP_BIN + ' point_triangulator --Mapper.ba_refine_principal_point 0 \
                                             --database_path {} \
                                             --image_path {} \
                                             --input_path {} \
                                             --output_path {} \
                                             --Mapper.filter_min_tri_angle 4.99 \
                                             --Mapper.init_max_forward_motion 1e20 \
                                             --Mapper.tri_min_angle 5.00 \
                                             --Mapper.tri_merge_max_reproj_error {} \
                                             --Mapper.tri_complete_max_reproj_error {} \
                                             --Mapper.filter_max_reproj_error {} \
                                             --Mapper.extract_colors 1 \
                                             --Mapper.ba_refine_focal_length 0 \
                                             --Mapper.ba_refine_extra_params 0\
                                             --Mapper.max_extra_param 1e20 \
                                             --Mapper.ba_local_num_images 6 \
                                             --Mapper.ba_local_max_num_iterations 100 \
                                             --Mapper.ba_global_images_ratio 1.0000001\
                                             --Mapper.ba_global_max_num_iterations 100 \
                                             --Mapper.tri_ignore_two_view_tracks 0'.format(db_file, img_dir, out_dir, out_dir,
                                                                                               tri_merge_max_reproj_error,
                                                                                               tri_complete_max_reproj_error,
                                                                                               filter_max_reproj_error)
    if debug_folder is not None:
        cmd += '  2>&1 | tee {}'.format(os.path.join(os.path.join(debug_folder, 'log_point_triangulation.txt')))
    bash_run(cmd)


def run_global_ba(in_dir, out_dir, weight, debug_folder=None):
    os.makedirs(out_dir, exist_ok=True)

    # global bundle adjustment
    # one meter is roughly three pixels, we should square it
    cmd = COLMAP_BIN + ' bundle_adjuster --input_path {in_dir} --output_path {out_dir} \
                                    --BundleAdjustment.max_num_iterations 5000 \
                                    --BundleAdjustment.refine_focal_length 0\
                                    --BundleAdjustment.refine_principal_point 1 \
                                    --BundleAdjustment.refine_extra_params 0 \
                                    --BundleAdjustment.refine_extrinsics 0 \
                                    --BundleAdjustment.function_tolerance 0 \
                                    --BundleAdjustment.gradient_tolerance 0 \
                                    --BundleAdjustment.parameter_tolerance 1e-10 \
                                    --BundleAdjustment.constrain_points 1 \
                                    --BundleAdjustment.constrain_points_loss_weight {weight}'.format(in_dir=in_dir, out_dir=out_dir, weight=weight)
    if debug_folder is not None:
        cmd += '  2>&1 | tee {}'.format(os.path.join(os.path.join(debug_folder, 'log_bundle_adjustment.txt')))

    bash_run(cmd)
