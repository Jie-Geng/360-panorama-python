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
    int idx;
    int row;
    int col;
    string uri;
    double pitch;
    double roll;
    double yaw;
    double x;
    double y;
    double width;
    double height;
} MetaRow;


typedef struct tag_MetaData{
    int nrows;
    int ncols;
    vector<MetaRow> files;
} MetaData;


// constants
#define BLEND_STRENGTH 5.0
#define SEAM_SCALE 0.3

// config data
vector<int> gv_row, gv_col, gv_x, gv_y;
vector<string> gv_image_name, gv_mask_name;

// program arguments
string gs_output, gs_config, gs_folder, gs_mode;

bool gb_verbose = false;

vector<Mat> gv_seam_masks;
vector<Mat> gv_images;
vector<int> gv_processed_indices;

#define LOG(msg) do{ if (gb_verbose) {std::cout << msg;} } while(false)
#define LOGLN(msg) LOG(msg << std::endl)

#define MSG(msg) std::cout << msg
#define MSGLN(msg) std::cout << msg << std::endl


static int parse_config_data()
{
    string config_file_path = gs_folder + "/" + gs_config;
    ifstream myfile (config_file_path);

    if (myfile.is_open())
    {
        vector<String> lines;
        string line;
        while ( getline (myfile, line) ) {
            lines.push_back(line);
        }

        int count = lines.size();

        gv_row.resize(count);
        gv_col.resize(count);
        gv_x.resize(count);
        gv_y.resize(count);
        gv_image_name.resize(count);
        gv_mask_name.resize(count);

        for(int i = 0; i < count; i++)
        {
            stringstream iss(lines[i]);

            if( gs_mode == "full" )
            {
                iss >> gv_image_name[i];
                iss >> gv_mask_name[i];
            }
            else
            {
                iss >> gv_row[i];
                iss >> gv_col[i];
            }

            iss >> gv_x[i];
            iss >> gv_y[i];
        }
    }
    else
    {
        MSGLN("ERROR: Cannot open meta file " << config_file_path);
        return -1;
    }

    return 0;
}


static void printUsage()
{
    cout <<
        "\nPanorama composer for image stitching. It reads images from the folder, compensates camera exposure, finds seams, blends image frames into one panorama, and stores it.\n\n"
        "USAGE: pano-composer OPTIONS\n\n"
        " --folder FOLDER <string>\n"
        "      Folder path that contains image, mask, and config files.\n\n"
        " --mode MODE <string> (frame|full)\n"
        "      Composition mode. [full] mode accept inputs indicating files. [frame] mode accept inputs indicating image grid data. \nIn [frame] mode, the result mask is stored in the FOLDER with the name of 'frame-mask.jpg'.\n\n"
        " --config CONFIG <string>\n"
        "      Config file name in the FOLDER.\n\n"
        " --output OUTPUT <string>\n"
        "      Result panorama file path to be stored.\n\n";
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
        else if (string(argv[i]) == "--folder")
        {
            if (i == argc-1){
                return -1;
            }
            gs_folder = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--config")
        {
            if (i == argc-1){
                return -1;
            }
            gs_config = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--mode")
        {
            if (i == argc-1){
                return -1;
            }
            gs_mode = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--output")
        {
            if (i == argc-1){
                return -1;
            }
            gs_output = argv[i + 1];
            i++;
        }
    }
 
    return 0;
}


int main(int argc, char* argv[])
{
    // test();
    // return 0;

    int64 app_start_time = getTickCount();

    ////////////////////////////////////////////////////////////////////////
    // Parsing and Loading
    ////////////////////////////////////////////////////////////////////////
    int retval = parseCmdArgs(argc, argv);
    if (retval)
    {
        MSGLN("\nERROR: Cannot parse parameters.\n");
        return retval;
    }

    // Check if neccessary parameters were given
    if (gs_config == "")
    {
        MSGLN("\nERROR: Config file name is not given.\n");
        return -1;
    }

    if (gs_folder == "")
    {
        MSGLN("\nERROR: Image folder path is not given.\n");
        return -1;
    }

    if (gs_mode == "")
    {
        MSGLN("\nERROR: Composition mode is not given.\n");
        return -1;
    }

    if (gs_output == "")
    {
        MSGLN("\nERROR: Output file name is not given.\n");
        return -1;
    }

    // Parse the meta file
    retval = parse_config_data();
    if (retval)
    {
        MSGLN("\nERROR: Cannot read the config file.\n");
        return retval;
    }

    ////////////////////////////////////////////////////////////////////////
    // Loading files
    int num_images = gv_x.size();

    vector<Point> corners(num_images);
    vector<UMat> masks(num_images);
    vector<UMat> images(num_images);
    vector<UMat> images_f(num_images);
    vector<Size> sizes(num_images);
    // vector<UMat> masks(num_images);
    vector<Point> seam_corners(num_images);
    vector<UMat> seam_masks(num_images);
    vector<UMat> seam_images(num_images);

    for(int i = 0; i < num_images; i++)
    {
        Mat raw_image;
        corners[i] = Point(gv_x[i], gv_y[i]);
        seam_corners[i] = Point(int(gv_x[i] * SEAM_SCALE), int(gv_y[i] * SEAM_SCALE));

        // read image file
        ostringstream stringStream;
        if( gs_mode == "frame" ){
            stringStream << gs_folder << "/warped-" << gv_row[i] << "-" << gv_col[i] << ".jpg";
        } else {
            stringStream << gs_folder << "/" << gv_image_name[i];
        }
        string file_name = stringStream.str();
        raw_image = imread(file_name);

        sizes[i] = Size(raw_image.cols, raw_image.rows);

        images[i] = raw_image.getUMat(ACCESS_READ);
        resize(images[i], seam_images[i], Size(), SEAM_SCALE, SEAM_SCALE, INTER_LINEAR_EXACT);
        seam_images[i].convertTo(images_f[i], CV_32F);

        // read mask file
        stringStream.str("");
        stringStream.clear();
        if( gs_mode == "frame" ){
            stringStream << gs_folder << "/mask-" << gv_row[i] << "-" << gv_col[i] << ".jpg";
        } else {
            stringStream << gs_folder << "/" << gv_mask_name[i];
        }
        file_name = stringStream.str();
        raw_image = imread(file_name);
        cvtColor(raw_image, masks[i], COLOR_BGR2GRAY);
        resize(masks[i], seam_masks[i], Size(), SEAM_SCALE, SEAM_SCALE, INTER_LINEAR_EXACT);
    }


    ////////////////////////////////////////////////////////////////////////
    // Compensating
    ////////////////////////////////////////////////////////////////////////
    int64 stage_start_time = getTickCount();
    LOGLN("\n-COMPENSATING EXPOSURES...........");

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
    BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
    bcompensator->setNrFeeds(1);
    bcompensator->setNrGainsFilteringIterations(2);
    bcompensator->setBlockSize(32, 32);
    compensator->feed(seam_corners, seam_images, seam_masks);

    LOGLN(" FINISHED in " << (getTickCount() - stage_start_time)/getTickFrequency() << " seconds.");


    ////////////////////////////////////////////////////////////////////////
    // Seam finding
    ////////////////////////////////////////////////////////////////////////
    stage_start_time = getTickCount();
    LOGLN("\n-SEAM FINDING...........");

    Ptr<SeamFinder> seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    seam_finder->find(images_f, seam_corners, seam_masks);

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

    Mat img_s;
    Mat dilated_mask, seam_mask, mask;

    for(size_t i = 0; i < images.size(); i++)
    {
        compensator->apply(i, corners[i], images[i], masks[i]);
        images[i].convertTo(img_s, CV_16S);

        dilate(seam_masks[i], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, masks[i].size(), 0, 0, INTER_LINEAR_EXACT);
        mask = seam_mask & (masks[i].getMat(ACCESS_READ));

        blender->feed(img_s, mask, corners[i]);        
    }    

    Mat result, result_mask;
    blender->blend(result, result_mask);
    LOGLN(" FINISHED in " << (getTickCount() - stage_start_time)/getTickFrequency() << " seconds.");


    ////////////////////////////////////////////////////////////////////////
    // Post process
    ////////////////////////////////////////////////////////////////////////
    MSGLN("FINISHED in " << (getTickCount() - app_start_time)/getTickFrequency() << " seconds.");

    imwrite(gs_output, result);

    if( gs_mode == "frame" ){
        // store mask result
        imwrite(gs_folder + "/frame-mask.jpg", result_mask);
    }

    return 0;
}
