#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;


typedef struct tag_MetaRow{
    int row;
    int col;
    string uri;
} MetaRow;


typedef struct tag_MetaData{
    int nrows;
    int ncols;
    vector<MetaRow> files;
} MetaData;


// constants
#define FEATURE_COUNT 750
#define MATCH_THRESHOLD 0.8
#define ADJUSTER_THRESHOLD 1.1
#define ADJUSTER_MAX_ITER 50
#define BLEND_STRENGTH 5.0

double gf_seam_megapix = 0.1;
double gf_compose_scale = 0.2;
float gf_conf_thresh = 1.f;
string gs_folder;
string gs_temp_folder;
string gs_meta_file_path;
double gf_scale = 0.4;
Ptr<MetaData> meta = NULL;

bool gb_verbose = false;
bool gb_show_images = false;  // true for only development

vector<ImageFeatures> gv_features;
vector<MatchesInfo> gv_matches;
vector<Mat> gv_images;
vector<int> gv_processed_indices;

int gn_roi_nrows, gn_roi_nimages, gn_roi_begin, gn_roi_end;

#define LOG(msg) do{ if (gb_verbose) {std::cout << msg;} } while(false)
#define LOGLN(msg) LOG(msg << std::endl)

#define MSG(msg) std::cout << msg
#define MSGLN(msg) std::cout << msg << std::endl

void test();


static Ptr<MetaData> parse_meta_data()
{
    Ptr<MetaData> meta_data = NULL;
    ifstream myfile (gs_meta_file_path);

    if (myfile.is_open())
    {
        vector<String> lines;
        string line;
        while ( getline (myfile, line) ) {
            lines.push_back(line);
        }

        int count = lines.size();
        meta_data = makePtr<MetaData>();

        meta_data->nrows = 0;
        meta_data->ncols = 0;

        meta_data->files.resize(count);

        for(int i = 0; i < count; i++)
        {
            stringstream iss(lines[i]);
            iss >> meta_data->files[i].row;
            iss >> meta_data->files[i].col;
            iss >> meta_data->files[i].uri;

            if (meta_data->files[i].row > meta_data->nrows){
                meta_data->nrows = meta_data->files[i].row;
            }
            if (meta_data->files[i].col > meta_data->ncols){
                meta_data->ncols = meta_data->files[i].col;
            }
        }

        meta_data->ncols++;
        meta_data->nrows++;
    }
    else
    {
        MSGLN("ERROR: Cannot open meta file " << gs_meta_file_path);
    }

    return meta_data;
}


static void printUsage()
{
    cout <<
        "\nPreprocess for image stitcher. It reads images from the folder according to the metadata, detects fatures, matches features, finds homography, warpes images, and stores them in the temporary folder.\n\n"
        "USAGE: prestitcher --folder FOLDER --meta META --temp-folder TEMPFOLDER --scale SCALE\n\n"
        " --meta META <string>\n"
        "      Meta file path.\n"
        " --folder FOLDER <string>\n"
        "      Folder path that contains the images files.\n"
        " --temp-folder TEMPFOLDER <string>\n"
        "      Folder path that temporary working files would be stored.\n"
        " --scale SCALE <float>\n"
        "      Scale to resize images when composing. The default is 1.0.\n"
        " --verbose\n"
        "      Print detailed messages.\n\n";
}


static int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage();
        return -1;
    }
 
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage();
            return -1;
        }
        else if (string(argv[i]) == "--meta")
        {
            if (i == argc-1){
                return 1;
            }
            gs_meta_file_path = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--folder")
        {
            if (i == argc-1){
                return 1;
            }
            gs_folder = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--temp-folder")
        {
            if (i == argc-1){
                return 1;
            }
            gs_temp_folder = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--scale")
        {
            if (i == argc-1){
                return 1;
            }
            gf_scale = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--verbose")
        {
            gb_verbose = true;
            i++;
        }
    }
 
    return 0;
}


static int find_features(int nfeatures = 500)
{
    // TWO ROWS
    gv_features.resize(gn_roi_nimages);
    gv_images.resize(gn_roi_nimages);
	
    Ptr<Feature2D> finder = ORB::create(nfeatures);
	Mat full_image, image, mask;
	int width, height;

    for (int i = 0; i < gn_roi_nimages; i++)
	{
        // TWO ROWS
		string name = gs_folder + "/" + meta->files[i + gn_roi_begin * meta->ncols].uri;
		full_image = imread(name);
        resize(full_image, image, Size(), gf_scale, gf_scale, INTER_LINEAR);
 
        if (image.empty())
        {
            MSGLN("ERROR: Can't open image " << name);
            return -1;
        }

        mask = Mat::zeros(image.size(), CV_8U);
        mask.setTo(Scalar::all(255));
        
        width = image.size().width;
        height = image.size().height;

        vector<KeyPoint> keypoints;
        vector<KeyPoint> sub_keypoints;
        UMat descriptors;

        // for middle rows, divide the ROI into 4 pieces top-left, top-right, bottom-left, and bottom-right quarters.
        mask.setTo(Scalar::all(0));
        mask(Range(0, height/2), Range(0, width/2)).setTo(Scalar::all(255));
        finder->detect(image, sub_keypoints, mask);
        keypoints.insert(keypoints.end(), sub_keypoints.begin(), sub_keypoints.end());

        mask.setTo(Scalar::all(0));
        mask(Range(0, height/2), Range(width/2, width-1)).setTo(Scalar::all(255));
        finder->detect(image, sub_keypoints, mask);
        keypoints.insert(keypoints.end(), sub_keypoints.begin(), sub_keypoints.end());

        mask.setTo(Scalar::all(0));
        mask(Range(height/2, height-1), Range(0, width/2)).setTo(Scalar::all(255));
        finder->detect(image, sub_keypoints, mask);
        keypoints.insert(keypoints.end(), sub_keypoints.begin(), sub_keypoints.end());

        mask.setTo(Scalar::all(0));
        mask(Range(height/2, height-1), Range(width/2, width-1)).setTo(Scalar::all(255));
        finder->detect(image, sub_keypoints, mask);
        keypoints.insert(keypoints.end(), sub_keypoints.begin(), sub_keypoints.end());

        finder->compute(image, keypoints, descriptors);

        gv_features[i].img_size = image.size();
        gv_features[i].keypoints = keypoints;
        gv_features[i].descriptors = descriptors;
        // computeImageFeatures(finder, image, features, mask);

        gv_features[i].img_idx = i;

        gv_images[i] = image.clone();

        if (gb_verbose)
            LOGLN("    Features in image #" << i+1 << ": " << gv_features[i].keypoints.size());

        if (gb_show_images)
        {
            Mat output;
            drawKeypoints(image, gv_features[i].keypoints, output, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            ostringstream stringStream;
            stringStream << "keypoints-" << meta->files[i].row << "-" << meta->files[i].col;
            string window_name = stringStream.str(); 

            namedWindow(window_name, WINDOW_NORMAL);
            resizeWindow(window_name, 600, 800);
            imshow(window_name, output);

            waitKey(0);
            destroyAllWindows();
        }

        full_image.release();
        image.release();
        mask.release();
	}

	return 0;
}


static void draw_matches()
{
    int count = 1;
    for(size_t i = 0; i < gv_matches.size(); i++){
        int src_idx = gv_matches[i].src_img_idx;
        int dst_idx = gv_matches[i].dst_img_idx;

        if (src_idx >= dst_idx)
            continue;

        vector<DMatch> matches_list = gv_matches[i].getMatches();
        if (matches_list.size() == 0)
            continue;

        vector<uchar> matches_mask = gv_matches[i].getInliers();
        char * ch_mask = (char*)matches_mask.data();
        vector<char> mask = vector<char>(ch_mask, ch_mask + matches_mask.size());
        
        Mat result_img;
        Mat src_img, dst_img;
        src_img = gv_images[src_idx];
        dst_img = gv_images[dst_idx];
        drawMatches(src_img, gv_features[src_idx].getKeypoints(),
                    dst_img, gv_features[dst_idx].getKeypoints(),
                    matches_list, result_img, Scalar(0, 255, 255), Scalar(255, 0, 0),
                    mask, DrawMatchesFlags::DEFAULT);
        
        int match_count = 0;
        for( size_t j = 0; j < matches_mask.size(); j++ ){
            match_count += matches_mask[j];
        }
        ostringstream stringStream;
        stringStream << "#" << count << "  Confidence=" << gv_matches[i].confidence << " Count=" << match_count;
        string window_name = stringStream.str(); 

        namedWindow(window_name, WINDOW_NORMAL);
        resizeWindow(window_name, 800, 600);
        imshow(window_name, result_img);

        waitKey(0);
        destroyAllWindows();

        count ++;
    }
}


static void generate_match_mask()
{
}


static int match_images()
{
    generate_match_mask();

    // Mat gm_match_mask;

    // // Making match mask
    // gm_match_mask = Mat::zeros(gn_roi_nimages, gn_roi_nimages, CV_8U);

    // for (int i = 0; i < gn_roi_nimages; i++)
    // {
    //     int row = i / meta->ncols;
    //     int col = i % meta->ncols;

    //     int pair_row, pair_col, pair_id;
    //     // same row 2 elements
    //     // same-row, prev-col
    //     pair_row = row;
    //     pair_col = col - 1;
    //     if (pair_col < 0) pair_col += meta->ncols;
    //     pair_id = pair_row * meta->ncols + pair_col;
    //     gm_match_mask.at<uchar>(i, pair_id) = 1;
    //     gm_match_mask.at<uchar>(pair_id, i) = 1;

    //     if( row < gn_roi_nrows)
    //     { // bottom 3 elements

    //         // // bottom-row, prev-col
    //         pair_row = row + 1;
    //         pair_col = col - 1;
    //         if (pair_col < 0) pair_col += meta->ncols;
    //         pair_id = pair_row * meta->ncols + pair_col;
    //         gm_match_mask.at<uchar>(i, pair_id) = 1;
    //         gm_match_mask.at<uchar>(pair_id, i) = 1;

    //         // bottom-row, same-col
    //         pair_col = col;
    //         pair_id = pair_row * meta->ncols + pair_col;
    //         gm_match_mask.at<uchar>(i, pair_id) = 1;
    //         gm_match_mask.at<uchar>(pair_id, i) = 1;

    //         // // bottom-row, next-col
    //         pair_col = col + 1;
    //         if (pair_col >= meta->ncols) pair_col -= meta->ncols;
    //         pair_id = pair_row * meta->ncols + pair_col;
    //         gm_match_mask.at<uchar>(i, pair_id) = 1;
    //         gm_match_mask.at<uchar>(pair_id, i) = 1;
    //     }

    // }

    // if (gb_verbose)
    // {
    //     LOGLN(" Match Mask:");
    //     for(int i = 0; i < gn_roi_nimages; i++)
    //     {
    //         if (i < 10) 
    //             LOG("   " << i << ": ");
    //         else
    //             LOG("  " << i << ": ");

    //         for(int j = 0; j < gn_roi_nimages; j++)
    //         {
    //             if (gm_match_mask.at<uchar>(i, j))
    //                 LOG("1");
    //             else
    //                 LOG(".");
    //         }
    //         LOGLN("");
    //     }
    //     LOGLN(" Mask End");
    // }

    Ptr<detail::FeaturesMatcher> matcher = makePtr<detail::BestOf2NearestMatcher>(false);
    (*matcher)(gv_features, gv_matches);
    // (*matcher)(gv_features, gv_matches, gm_match_mask.getUMat(ACCESS_READ));
    matcher->collectGarbage();

    if (gb_show_images)
        draw_matches();

    return 0;
}


static vector<CameraParams> adjust_camera_params()
{
    vector<ImageFeatures> original_features = gv_features;
    vector<MatchesInfo> original_matches = gv_matches;

    double threshold = MATCH_THRESHOLD;

    vector<int> indices;
    int64 start_time;
    int64 total_start_time = getTickCount();
    LOGLN("\n-OPTIMIZING CAMERA PARAMS...........");

    Ptr<Estimator> estimator = makePtr<HomographyBasedEstimator>();

    Ptr<detail::BundleAdjusterBase> adjuster = makePtr<detail::BundleAdjusterRay>();
    adjuster->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, ADJUSTER_MAX_ITER, 1e-2));
    adjuster->setConfThresh(ADJUSTER_THRESHOLD);

    vector<CameraParams> cameras;

    while(true)
    {
        gv_features = original_features;
        gv_matches = original_matches;

        start_time = getTickCount();
        LOG("   Trying with threshold " << threshold);
        indices = leaveBiggestComponent(gv_features, gv_matches, threshold);
        LOG("(" << indices.size() << " images) .........");

        // Estimating Camera params
        if (!(*estimator)(gv_features, gv_matches, cameras))
        {
            MSG("Homography estimation failed.");
            return vector<CameraParams>();
        }

        for (size_t i = 0; i < cameras.size(); ++i)
        {
            Mat R;
            cameras[i].R.convertTo(R, CV_32F);
            cameras[i].R = R;
        }


        // Adjusting camera params
        if (!(*adjuster)(gv_features, gv_matches, cameras))
        {
            LOGLN(" failed in " << (getTickCount() - start_time)/getTickFrequency() << " seconds.");
            threshold += 0.1;
            continue;
        }

        LOGLN(" success in " << (getTickCount() - start_time)/getTickFrequency() << " seconds.");
        
        LOGLN(" FINISHED in " << (getTickCount() - total_start_time)/getTickFrequency() << " seconds.");
        break;
    }

    gv_processed_indices = indices;
    return cameras;
}


int main(int argc, char* argv[])
{
    // test();
    // return 0;

    int64 app_start_time = getTickCount();


    ////////////////////////////////////////////////////////////////////////
    // Parsing parameters
    int retval = parseCmdArgs(argc, argv);
    if (retval)
    {
        if (retval > 0){
            MSGLN("\nERROR: parameters are not given.\n");
        }
        return retval;
    }

    // Check if neccessary parameters were given
    if (gs_meta_file_path == "")
    {
        MSGLN("\nERROR: Meta file path is not given.\n");
        return -1;
    }

    if (gs_folder == "")
    {
        MSGLN("\nERROR: Image folder path is not given.\n");
        return -1;
    }

    if (gs_temp_folder == "")
    {
        MSGLN("\nERROR: Temp folder is not given.\n");
        return -1;
    }


    ////////////////////////////////////////////////////////////////////////
    // Parse the meta file
    meta = parse_meta_data();
    if( meta )
    {
    	if (gb_verbose)
    	{
	        LOGLN("-META DATA:");
	        LOGLN("    Row Count: " << meta->nrows);
	        LOGLN("    Col Count: " << meta->ncols);

	        for(unsigned int idx = 0; idx < meta->files.size(); idx++)
	        {
	            LOGLN("    " << meta->files[idx].row << "-" << meta->files[idx].col << ": " << meta->files[idx].uri);
	        }
    	}
    }
    else
    {
        LOGLN("\nERROR: Cannot parse meta data.\n");
        return 0;
    }


    ////////////////////////////////////////////////////////////////////////
    // Choose what rows to deal with
    ////////////////////////////////////////////////////////////////////////
    switch (meta->nrows)
    {
        case 4:
            gn_roi_nrows = 2;
            gn_roi_begin = 1;
            gn_roi_begin = 2;
            break;
        case 5:
            gn_roi_nrows = 2;
            gn_roi_begin = 2;
            gn_roi_end = 3;
            break;
        case 6:
            gn_roi_nrows = 2;
            gn_roi_begin = 2;
            gn_roi_end = 3;
            break;
        case 7:
            gn_roi_nrows = 3;
            gn_roi_begin = 2;
            gn_roi_end = 4;
            break;
        case 8:
            gn_roi_nrows = 4;
            gn_roi_begin = 2;
            gn_roi_end = 5;
            break;
        case 9:
            gn_roi_nrows = 4;
            gn_roi_begin = 3;
            gn_roi_end = 6;
            break;
        default:
            MSGLN("\nERROR: number of rows should be in 4 ~ 9.\n");
            return -1;
    }
    gn_roi_nimages = gn_roi_nrows * meta->ncols;


    ////////////////////////////////////////////////////////////////////////
    // Find features
    ////////////////////////////////////////////////////////////////////////
    int64 stage_start_time = getTickCount();
    LOGLN("\n-FINDING FEATURES...........");
    
    int ret = find_features(FEATURE_COUNT);
    if (ret)
    {
        return -1;
    }
    LOGLN(" finished in " << (getTickCount() - stage_start_time)/getTickFrequency() << " seconds.");


    ////////////////////////////////////////////////////////////////////////
    // Match images
    ////////////////////////////////////////////////////////////////////////
    stage_start_time = getTickCount();
    LOGLN("\n-MATCHING IMAGES...........");
    ret = match_images();
    if (ret)
    {
        return -1;
    }
    LOGLN(" finished in " << (getTickCount() - stage_start_time)/getTickFrequency() << " seconds.");


    ////////////////////////////////////////////////////////////////////////
    // Bundle adjustment
    ////////////////////////////////////////////////////////////////////////
    vector<CameraParams> cameras = adjust_camera_params();

    // Wave correction
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        LOGLN("Camera #" << gv_processed_indices[i] << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());

    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    LOGLN("\nWapred Image Scale is" << warped_image_scale);

    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R.clone());
    waveCorrect(rmats, detail::WAVE_CORRECT_HORIZ);
    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].R = rmats[i];


    ////////////////////////////////////////////////////////////////////////
    // Warping images
    ////////////////////////////////////////////////////////////////////////
    Ptr<WarperCreator> warper_creator = makePtr<cv::SphericalWarper>();
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * gf_scale));

    int num_images = gv_processed_indices.size();

    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<UMat> images_warped_f(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);

    stage_start_time = getTickCount();
    LOGLN("\n-WARPING & SAVING IMAGES...........");
    for(int i = 0; i < num_images; i++)
    {
        // warp image
        masks[i].create(gv_images[gv_processed_indices[i]].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));

        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        // K(0, 0) *= gf_scale; K(0, 2) *= gf_scale;
        // K(1, 1) *= gf_scale; K(1, 2) *= gf_scale;

        corners[i] = warper->warp(gv_images[gv_processed_indices[i]], 
            K, cameras[i].R, INTER_NEAREST, BORDER_REFLECT, images_warped[i]);
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, 
            masks_warped[i]);
        sizes[i] = images_warped[i].size();

        images_warped[i].convertTo(images_warped_f[i], CV_32F);

        // save image
        int idx = gv_processed_indices[i] + gn_roi_begin * meta->ncols;
        ostringstream stringStream;
        stringStream << gs_temp_folder << "/warped-" << meta->files[idx].row << "-" << meta->files[idx].col << ".jpg";

        string file_name = stringStream.str();
        LOGLN("    " << file_name);
        imwrite(file_name, images_warped[i]);

        stringStream.str("");
        stringStream.clear();
        stringStream << gs_temp_folder << "/mask-" << meta->files[idx].row << "-" << meta->files[idx].col << ".jpg";
        file_name = stringStream.str();
        LOGLN("    " << file_name);
        imwrite(file_name, masks_warped[i]);
    }

    // writing corners
    LOGLN("    Writing corners");
    ofstream datafile;
    datafile.open(gs_temp_folder + "/framedata");
    if( datafile.is_open() )
    {
        for(int i = 0; i < num_images; i++)
        {
            int idx = gv_processed_indices[i] + gn_roi_begin * meta->ncols;
            datafile << meta->files[idx].row << " " << 
                        meta->files[idx].col  << " " << 
                        corners[i].x << " " << 
                        corners[i].y << " " << 
                        sizes[i].width << " " << 
                        sizes[i].height << std::endl;
        }
        datafile.close();
    }
    LOGLN(" finished in " << (getTickCount() - stage_start_time)/getTickFrequency() << " seconds.");

    
    cout << "\nFINISHED in " << (getTickCount() - app_start_time)/getTickFrequency() << " seconds.\n\n";

    return 0;

    // It's OK for this util and the left parts are for the next one.

    ////////////////////////////////////////////////////////////////////////
    // Compensating
    ////////////////////////////////////////////////////////////////////////
    stage_start_time = getTickCount();
    LOGLN("\n-COMPENSATING EXPOSURES...........");

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
    BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
    bcompensator->setNrFeeds(1);
    bcompensator->setNrGainsFilteringIterations(2);
    bcompensator->setBlockSize(32, 32);
    compensator->feed(corners, images_warped, masks_warped);

    LOGLN(" FINISHED in " << (getTickCount() - stage_start_time)/getTickFrequency() << " seconds.");


    ////////////////////////////////////////////////////////////////////////
    // Seam finding
    ////////////////////////////////////////////////////////////////////////
    stage_start_time = getTickCount();
    LOGLN("\n-SEAM FINDING...........");

    Ptr<SeamFinder> seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    seam_finder->find(images_warped_f, corners, masks_warped);

    LOGLN(" FINISHED in " << (getTickCount() - stage_start_time)/getTickFrequency() << " seconds.");


    ////////////////////////////////////////////////////////////////////////
    // Blending
    ////////////////////////////////////////////////////////////////////////
    stage_start_time = getTickCount();
    LOGLN("\n-BLENDING...........");

    Ptr<Blender> blender = Blender::createDefault(Blender::MULTI_BAND, false);
    Size dst_sz = resultRoi(corners, sizes).size();
    float blend_width = sqrt(static_cast<float>(dst_sz.area())) * BLEND_STRENGTH / 100.f;
    MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
    int num_bands = static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.);
    mb->setNumBands(num_bands);
    blender->prepare(corners, sizes);

    Mat img_warped_s;

    for(size_t i = 0; i < images_warped.size(); i++)
    {

        compensator->apply(i, corners[i], images_warped[i], masks_warped[i]);
        images_warped[i].convertTo(img_warped_s, CV_16S);
        blender->feed(img_warped_s, masks_warped[i], corners[i]);        
    }    

    Mat result, result_mask;
    blender->blend(result, result_mask);
    LOGLN(" FINISHED in " << (getTickCount() - stage_start_time)/getTickFrequency() << " seconds.");


    ////////////////////////////////////////////////////////////////////////
    // Post process
    ////////////////////////////////////////////////////////////////////////
    LOGLN("\nALL PROCESS FINISHED in " << (getTickCount() - app_start_time)/getTickFrequency() << " seconds.");

    imwrite("/tmp/result.jpg", result);
    return 0;
}

void test()
{
    Mat img_warped;
    Mat img = imread("/panorama/images/recent-04/img-r1-000.jpg");
    Size sz = img.size();
    Mat K = Mat::zeros(Size(3, 3), CV_32F);
    double focal = 1100.21;
    K.at<double>(0, 0) = focal; K.at<double>(1, 1) = focal;
    K.at<double>(0, 2) = sz.width / 2.0;
    K.at<double>(1, 2) = sz.height / 2.0;

    // Calculate rotation about x axis
    double theta_x = 0.1;
    double theta_y = 0;
    double theta_z = 0;
    Mat R_x = (Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta_x),   -sin(theta_x),
               0,       sin(theta_x),   cos(theta_x)
               );
     
    // Calculate rotation about y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(theta_y),    0,      sin(theta_y),
               0,               1,      0,
               -sin(theta_y),   0,      cos(theta_y)
               );
     
    // Calculate rotation about z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(theta_z),    -sin(theta_z),      0,
               sin(theta_z),    cos(theta_z),       0,
               0,               0,                  1);
     
     
    // Combined rotation matrix
    Mat R_exp = R_z * R_y * R_x;
    Mat R;
    R_exp.convertTo(R, CV_32F);

    Ptr<WarperCreator> warper_creator = makePtr<cv::SphericalWarper>();
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(focal));

    warper->warp(img, K, R, INTER_NEAREST, BORDER_CONSTANT, img_warped);

    namedWindow("original", WINDOW_NORMAL);
    resizeWindow("original", 600, 800);
    namedWindow("warped", WINDOW_NORMAL);
    resizeWindow("warped", 600, 800);
    imshow("original", img);
    imshow("warped", img_warped);
    waitKey(0);
    destroyAllWindows();

}