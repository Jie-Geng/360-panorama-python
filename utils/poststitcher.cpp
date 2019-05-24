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

double gf_seam_scale = 0.4;
string gs_temp_folder;
Ptr<MetaData> meta = NULL;

bool gb_verbose = false;
bool gb_show_images = false;  // true for only development

vector<Mat> gv_seam_masks;
vector<Mat> gv_images;
vector<int> gv_processed_indices;

#define LOG(msg) do{ if (gb_verbose) {std::cout << msg;} } while(false)
#define LOGLN(msg) LOG(msg << std::endl)

#define MSG(msg) std::cout << msg
#define MSGLN(msg) std::cout << msg << std::endl


static Ptr<MetaData> parse_meta_data()
{
    Ptr<MetaData> meta_data = NULL;
    string meta_file_path = gs_temp_folder + "/framedata.all";
    ifstream myfile (meta_file_path);

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
            iss >> meta_data->files[i].idx;
            iss >> meta_data->files[i].row;
            iss >> meta_data->files[i].col;
            iss >> meta_data->files[i].uri;
            iss >> meta_data->files[i].pitch;
            iss >> meta_data->files[i].roll;
            iss >> meta_data->files[i].yaw;
            iss >> meta_data->files[i].x;
            iss >> meta_data->files[i].y;
            iss >> meta_data->files[i].width;
            iss >> meta_data->files[i].height;

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
        MSGLN("ERROR: Cannot open meta file " << meta_file_path);
    }

    return meta_data;
}


static void printUsage()
{
    cout <<
        "\nPost-process for image stitcher. It reads images from the temporary folder, compenstes camera exposure, finds seams, blends image frames into one panorama, and stores it.\n\n"
        "USAGE: poststitcher --temp-folder TEMPFOLDER\n\n"
        " --temp-folder TEMPFOLDER <string>\n"
        "      Folder path that temporary working files would be stored.\n\n";
}


static int parseCmdArgs(int argc, char** argv)
{
    if (argc != 3)
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
        else if (string(argv[i]) == "--temp-folder")
        {
            if (i == argc-1){
                return 1;
            }
            gs_temp_folder = argv[i + 1];
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
        if (retval > 0){
            MSGLN("\nERROR: parameters are not given.\n");
        }
        return retval;
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
                LOGLN("    " << meta->files[idx].row << "-" << meta->files[idx].col << ": (" << meta->files[idx].x << ", " << meta->files[idx].y << ") (" << meta->files[idx].width << ", " << meta->files[idx].height << ")");
            }
        }
    }
    else
    {
        LOGLN("\nERROR: Cannot parse meta data.\n");
        return 0;
    }

    ////////////////////////////////////////////////////////////////////////
    // Loading files
    int num_images = meta->files.size();

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
        corners[i] = Point(int(meta->files[i].x), int(meta->files[i].y));
        sizes[i] = Size(int(meta->files[i].width), int(meta->files[i].height));
        seam_corners[i] = Point(int(meta->files[i].x * gf_seam_scale), int(meta->files[i].y * gf_seam_scale));

        // read image file
        ostringstream stringStream;
        stringStream << gs_temp_folder << "/warped-" << meta->files[i].row << "-" << meta->files[i].col << ".jpg";
        string file_name = stringStream.str();
        raw_image = imread(file_name);
        images[i] = raw_image.getUMat(ACCESS_READ);
        resize(images[i], seam_images[i], Size(), gf_seam_scale, gf_seam_scale, INTER_LINEAR_EXACT);
        seam_images[i].convertTo(images_f[i], CV_32F);

        // read mask file
        stringStream.str("");
        stringStream.clear();
        stringStream << gs_temp_folder << "/mask-" << meta->files[i].row << "-" << meta->files[i].col << ".jpg";
        file_name = stringStream.str();
        raw_image = imread(file_name);
        cvtColor(raw_image, masks[i], COLOR_BGR2GRAY);
        resize(masks[i], seam_masks[i], Size(), gf_seam_scale, gf_seam_scale, INTER_LINEAR_EXACT);
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
    LOGLN("\nALL PROCESS FINISHED in " << (getTickCount() - app_start_time)/getTickFrequency() << " seconds.");

    imwrite("/tmp/post-result.jpg", result);
    return 0;
}
