/*
 * Copyright (C) 2018  Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cluon-complete.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>

int32_t main(int32_t argc, char **argv) {
    int32_t retCode{1};
    auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if ( (0 == commandlineArguments.count("name")) ||
         (0 == commandlineArguments.count("width")) ||
         (0 == commandlineArguments.count("height")) ) {
        std::cerr << argv[0] << " attaches to a shared memory area containing an ARGB image and transform it to HSV color space for inspection." << std::endl;
        std::cerr << "Usage:   " << argv[0] << " --name=<name of shared memory area> --width=<W> --height=<H>" << std::endl;
        std::cerr << "         --name:   name of the shared memory area to attach" << std::endl;
        std::cerr << "         --width:  width of the frame" << std::endl;
        std::cerr << "         --height: height of the frame" << std::endl;
        std::cerr << "Example: " << argv[0] << " --name=img.argb --width=640 --height=480" << std::endl;
    }
    else {
        const std::string NAME{commandlineArguments["name"]};
        const uint32_t WIDTH{static_cast<uint32_t>(std::stoi(commandlineArguments["width"]))};
        const uint32_t HEIGHT{static_cast<uint32_t>(std::stoi(commandlineArguments["height"]))};

        // Attach to the shared memory.
        std::unique_ptr<cluon::SharedMemory> sharedMemory{new cluon::SharedMemory{NAME}};
        if (sharedMemory && sharedMemory->valid()) {
            std::clog << argv[0] << ": Attached to shared memory '" << sharedMemory->name() << " (" << sharedMemory->size() << " bytes)." << std::endl;

            // Create an OpenCV image header using the data in the shared memory.
            IplImage *iplimage{nullptr};
            CvSize size;
            size.width = WIDTH;
            size.height = HEIGHT;

            iplimage = cvCreateImageHeader(size, IPL_DEPTH_8U, 4 /* four channels: ARGB */);
            sharedMemory->lock();
            {
                iplimage->imageData = sharedMemory->data();
                iplimage->imageDataOrigin = iplimage->imageData;
            }
            sharedMemory->unlock();

            cv::namedWindow("Inspector", CV_WINDOW_AUTOSIZE);
            int minH{0};
            int maxH{179};
            cvCreateTrackbar("Hue (min)", "Inspector", &minH, 179);
            cvCreateTrackbar("Hue (max)", "Inspector", &maxH, 179);

            int minS{0};
            int maxS{255};
            cvCreateTrackbar("Sat (min)", "Inspector", &minS, 255);
            cvCreateTrackbar("Sat (max)", "Inspector", &maxS, 255);

            int minV{0};
            int maxV{255};
            cvCreateTrackbar("Val (min)", "Inspector", &minV, 255);
            cvCreateTrackbar("Val (max)", "Inspector", &maxV, 255);

            int hAdd{0};
            int sAdd{0};
            int vAdd{0};
            cvCreateTrackbar("Hadd", "Inspector", &hAdd, 179);
            cvCreateTrackbar("Sadd", "Inspector", &sAdd, 255);
            cvCreateTrackbar("Vadd", "Inspector", &vAdd, 255);

            int hSub{0};
            int sSub{0};
            int vSub{0};
            cvCreateTrackbar("Hsub", "Inspector", &hSub, 179);
            cvCreateTrackbar("Ssub", "Inspector", &sSub, 255);
            cvCreateTrackbar("Vsub", "Inspector", &vSub, 255);

            // Endless loop; end the program by pressing Ctrl-C.
            while (cv::waitKey(10)) {
                cv::Mat img;

                // Don't wait for a notification of a new frame so that the sender can pause while we are still inspection
                //sharedMemory->wait();

                // Lock the shared memory.
                sharedMemory->lock();
                {
                    // Copy image into cvMat structure.
                    // Be aware of that any code between lock/unlock is blocking
                    // the camera to provide the next frame. Thus, any
                    // computationally heavy algorithms should be placed outside
                    // lock/unlock.
                    img = cv::cvarrToMat(iplimage);
                }
                sharedMemory->unlock();

                cv::Mat imgHSV;
                cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

                std::vector<cv::Mat> hsvChannels;
                cv::split(imgHSV, hsvChannels);

                for (int i = 0; i < 3; ++i) {
                    hsvChannels[i].convertTo(hsvChannels[i], CV_16S);
                }

                hsvChannels[0] = cv::min(cv::max(hsvChannels[0] - hSub, 0) + hAdd, 179);
                hsvChannels[1] = cv::min(cv::max(hsvChannels[1] - sSub, 0) + sAdd, 255);
                hsvChannels[2] = cv::min(cv::max(hsvChannels[2] - vSub, 0) + vAdd, 255);

                for (int i = 0; i < 3; ++i) {
                    hsvChannels[i].convertTo(hsvChannels[i], CV_8U);
                }

                cv::merge(hsvChannels, imgHSV);

                // Create a mask using inRange
                cv::Mat mask;
                cv::inRange(imgHSV, cv::Scalar(minH, minS, minV), cv::Scalar(maxH, maxS, maxV), mask);

                // Convert adjusted HSV back to BGR for display
                cv::Mat adjustedBGR;
                cv::cvtColor(imgHSV, adjustedBGR, cv::COLOR_HSV2BGR);

                // Apply mask to show only matching areas
                cv::Mat filtered;
                adjustedBGR.copyTo(filtered, mask);

                cv::imshow("Mask only", mask);
                cv::imshow("Adjusted and masked", filtered);
            }

            if (nullptr != iplimage) {
                cvReleaseImageHeader(&iplimage);
            }
        }
        retCode = 0;
    }
    return retCode;
}

