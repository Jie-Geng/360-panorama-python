import cv2 as cv
import math


class CVStitcher:

    def __init__(self):
        self.register_resol_ = None  # float
        self.seam_est_resol_ = None  # float
        self.compose_resol_ = None  # float
        self.conf_thresh_ = None  # float
        self.interp_flags_ = None  # int
        self.features_finder_ = None  # object
        self.features_matcher_ = None  # object
        self.matching_mask_ = None  # matrix
        self.bundle_adjuster_ = None  # object
        self.estimator_ = None  # object
        self.do_wave_correct_ = None  # bool
        self.wave_correct_kind_ = None  # int
        self.warper_ = None  # object
        self.exposure_comp_ = None  # object
        self.seam_finder_ = None  # object
        self.blender_ = None  # object

        self.imgs_ = []
        self.masks_ = []
        self.full_img_sizes_ = []
        self.features_ = []
        self.pairwise_matches_ = []
        self.seam_est_imgs_ = []
        self.indices_ = []
        self.cameras_ = []
        self.result_mask_ = None  # matrix
        self.work_scale_ = None  # float
        self.seam_scale_ = None  # float
        self.seam_work_aspect_ = None  # float
        self.warped_image_scale_ = None  # float

    def registration_resol(self):
        return self.register_resol_

    def set_registration_resol(self, resol_mpx):
        self.register_resol_ = resol_mpx

    def seam_estimation_resol(self):
        return self.seam_est_resol_

    def set_seam_estimation_resol(self, resol_mpx):
        self.seam_est_resol_ = resol_mpx

    def compositing_resol(self):
        return self.compose_resol_

    def set_compositing_resol(self, resol_mpx):
        self.compose_resol_ = resol_mpx

    def pano_confidence_thresh(self):
        return self.conf_thresh_

    def set_pano_confidence_thresh(self, conf_thresh):
        self.conf_thresh_ = conf_thresh

    def wave_correction(self):
        return self.do_wave_correct_

    def set_wave_correction(self, flag):
        self.do_wave_correct_ = flag

    def interpolation_flags(self):
        return self.interp_flags_

    def set_interpolation_flags(self, interp_flags):
        self.interp_flags_ = interp_flags

    def wave_correct_kind(self):
        return self.wave_correct_kind_

    def set_wave_correct_kind(self, kind):
        self.wave_correct_kind_ = kind

    def features_finder(self):
        return self.features_finder_

    def set_features_finder(self, features_finder):
        self.features_finder_ = features_finder

    def features_matcher(self):
        return self.features_matcher_

    def set_features_matcher(self, features_matcher):
        self.features_matcher_ = features_matcher

    def matching_mask(self):
        return self.matching_mask_

    def set_matching_mask(self, mask):
        # CV_Assert(mask.type() == CV_8U && mask.cols == mask.rows);
        self.matching_mask_ = mask.copy()

    def bundle_adjuster(self):
        return self.bundle_adjuster_

    def set_bundle_adjuster(self, bundle_adjuster):
        self.bundle_adjuster_ = bundle_adjuster

    def estimator(self):
        return self.estimator_

    def set_estimator(self, estimator):
        self.estimator_ = estimator

    def warper(self):
        return self.warper_

    def set_warper(self, creator):
        self.warper_ = creator

    def exposure_compensator(self):
        return self.exposure_comp_

    def set_exposure_compensator(self, exposure_comp):
        self. exposure_comp_ = exposure_comp

    def seam_finder(self):
        return self.seam_finder_

    def set_seam_finder(self, seam_finder):
        self.seam_finder_ = seam_finder

    def blender(self):
        return self.blender_

    def set_blender(self, b):
        self.blender_ = b

    @staticmethod
    def create():
        stitcher = CVStitcher()

        stitcher.set_registration_resol(0.6)
        stitcher.set_seam_estimation_resol(0.1)
        stitcher.set_compositing_resol(-1.0)
        stitcher.set_pano_confidence_thresh(1)
        stitcher.set_seam_finder(cv.detail_GraphCutSeamFinder("COST_COLOR"))
        stitcher.set_blender(cv.detail_MultiBandBlender(False))
        stitcher.set_features_finder(cv.ORB())
        stitcher.set_interpolation_flags(cv.INTER_LINEAR_EXACT)

        stitcher.work_scale_ = 1
        stitcher.seam_scale_ = 1
        stitcher.seam_work_aspect_ = 1
        stitcher.warped_image_scale_ = 1

        stitcher.set_estimator(cv.detail_HomographyBasedEstimator())
        stitcher.set_wave_correction(True)
        stitcher.set_wave_correct_kind(cv.detail.WAVE_CORRECT_HORIZ)
        stitcher.set_features_matcher(cv.detail.BestOf2NearestMatcher_create(False))
        stitcher.set_bundle_adjuster(cv.detail_BundleAdjusterRay())
        stitcher.set_warper(cv.PyRotationWarper('spherical',
                                                stitcher.warped_image_scale_ * stitcher.seam_work_aspect_))
        stitcher.set_exposure_compensator(cv.detail_BlocksGainCompensator())

        return stitcher

    def estimate_transform(self, images, masks):
        self.imgs_ = images
        self.masks_ = masks

        ret = self.match_images()
        if ret != 0:
            return ret

        ret = self.estimate_camera_params()
        if ret != 0:
            return ret

        return 0

    def match_images(self):

        if len(self.imgs_) < 2:
            print("Need more images.")
            return -1

        self.work_scale_ = 1
        self.seam_work_aspect_ = 1
        self.seam_scale_ = 1

        is_work_scale_set = False
        # is_seam_scale_set = False

        feature_find_imgs = []
        # feature_find_masks = []

        for img in self.imgs_:
            h, w = img.shape[1], img.shape[0]
            self.full_img_sizes_.append((w, h))

            if self.register_resol_ < 0:
                feature_find_imgs.append(img)
                self.work_scale_ = 1
                is_work_scale_set = True
            else:
                if not is_work_scale_set:
                    self.work_scale_ = min(1.0, math.sqrt(self.register_resol_ * 1e6 / (w * h)))
                    is_work_scale_set = True

    def estimate_camera_params(self):
        if self.work_scale_ == 0:
            return 0
        return 0
