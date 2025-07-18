/*
 * Copyright (c) 2013, Marc Lebrun <marc.lebrun.ik@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <string>
#include <sstream>

#include "Utilities/Utilities.h"
#include "NlBayes/NlBayes.h"
#include "Utilities/LibImages.h"

using namespace std;

/**
 * @file   main.cpp
 * @brief  Main executable file
 *
 *
 *
 * @author MARC LEBRUN  <marc.lebrun.ik@gmail.com>
 **/

int main(int argc, char **argv)
{
    //! Check if there is the right call for the algorithm
	if (argc < 14) {
		cout << "usage: NL_Bayes image sigma add_noise noisy denoised basic difference \
		bias basic_bias diff_bias useArea1 useArea2 computeBias" << endl;
		return EXIT_FAILURE;
	}

    //! Variables initialization
	const float sigma   = atof(argv[2]);
	int add_noise = atoi(argv[3]);
	const bool doBias   = (bool) atof(argv[13]);
	const bool useArea1 = (bool) atof(argv[11]);
	const bool useArea2 = (bool) atof(argv[12]);
	const bool verbose  = true;

	//! Declarations
	vector<float> im, imNoisy, imBasic, imFinal, imDiff;
	vector<float> imBias, imBasicBias, imDiffBias;
	ImageSize imSize;

    //! Load image
	if(loadImage(argv[1], im, imSize, verbose) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
	}


	//! Add noise
	if (add_noise) {
		addNoise(im, imNoisy, sigma, verbose);
	 }
	else {
		imNoisy = im;
	}

    //! Denoising
    if (verbose) {
        cout << endl << "Applying NL-Bayes to the noisy image :" << endl;
    }
    if (runNlBayes(imNoisy, imBasic, imFinal, imSize, useArea1, useArea2, sigma, verbose)
        != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (verbose) {
        cout << endl;
    }

    //! Bias denoising
	if (doBias) {
        if (verbose) {
            cout << "Applying NL-Bayes to the original image :" << endl;
        }
        if (runNlBayes(im, imBasicBias, imBias, imSize, useArea1, useArea2, sigma, verbose)
             != EXIT_SUCCESS) {
             return EXIT_FAILURE;
        }
        if (verbose) {
            cout << endl;
        }
	}

	//! Compute PSNR and RMSE
    float psnr, rmse, psnrBasic, rmseBasic;
    computePsnr(im, imBasic, psnrBasic, rmseBasic, "imBasic", verbose);
    computePsnr(im, imFinal, psnr, rmse, "imFinal", verbose);

    float psnrBias, psnrBiasBasic, rmseBias, rmseBiasBasic;
    if (doBias) {
        computePsnr(im, imBasicBias, psnrBiasBasic, rmseBiasBasic, "imBiasBasic", verbose);
        computePsnr(im, imBias, psnrBias, rmseBias, "imBiasFinal", verbose);
    }

    //! writing measures
    writingMeasures("measures.txt", sigma, psnrBasic, rmseBasic, true, "_basic");
    writingMeasures("measures.txt", sigma, psnr, rmse, false, "      ");
    if (doBias) {
        writingMeasures("measures.txt", sigma, psnrBiasBasic, rmseBiasBasic, false, "_bias_basic");
        writingMeasures("measures.txt", sigma, psnrBias, rmseBias, false, "_bias      ");
    }

    //! Compute Difference
	if (computeDiff(im, imFinal, imDiff, sigma, 0.f, 255.f, verbose) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
	}
	if (doBias) {
        if (computeDiff(im, imBias, imDiffBias, sigma, 0.f, 255.f, verbose) != EXIT_SUCCESS) {
            return EXIT_FAILURE;
        }
	}

    //! save noisy, denoised and differences images
	if (verbose) {
	    cout << "Save images...";
	}
	if (add_noise) {
		if (saveImage(argv[4], imNoisy, imSize, 0.f, 255.f) != EXIT_SUCCESS) {
			return EXIT_FAILURE; }
	}

	if (saveImage(argv[5], imFinal, imSize, 0.f, 255.f) != EXIT_SUCCESS) {
		return EXIT_FAILURE;
	}

    if (saveImage(argv[6], imBasic, imSize, 0.f, 255.f) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    if (saveImage(argv[7], imDiff, imSize, 0.f, 255.f) != EXIT_SUCCESS) {
		return EXIT_FAILURE;
    }

    if (doBias) {
        if (saveImage(argv[8], imBias, imSize, 0.f, 255.f) != EXIT_SUCCESS) {
            return EXIT_FAILURE;
        }

        if (saveImage(argv[9], imBasicBias, imSize, 0.f, 255.f) != EXIT_SUCCESS) {
            return EXIT_FAILURE;
        }

        if (saveImage(argv[10], imDiffBias, imSize, 0.f, 255.f) != EXIT_SUCCESS) {
            return EXIT_FAILURE;
        }
    }
    if (verbose) {
        cout << "done." << endl;
    }

	return EXIT_SUCCESS;
}
