#include <stdio.h>

float inline sigmoid(float x) {
    return (x >= 0) ? x : 0;
}

void conv2d(float *data, int dataSize, int dataChannel,
            float *filters, int filterSize, int nFilters,
            float *biases, int stride,
            float *output)
{
    int iFilter, iRow, iCol, iFilterRow, iFilterCol,iChannel;
    int outputSize = (dataSize - filterSize) / stride + 1;
    float value;

    int cLength = dataSize * dataSize;
    int kLength = filterSize * filterSize;
    int kcount = kLength * dataChannel;

    #pragma omp parallel for default(shared) private(iFilter,iRow, iCol, iFilterRow, iFilterCol, value)
    for (iFilter = 0; iFilter < nFilters; iFilter++) {
            for (iRow = 0; iRow <  outputSize; iRow++) {
            for (iCol = 0; iCol < outputSize; iCol++) {
                //this is each of the outputs of the convolution. Each pixel in each output channel
                value = biases[iFilter];
                for (iChannel = 0; iChannel < dataChannel ; iChannel++)
                    for (iFilterRow = 0; iFilterRow < filterSize; iFilterRow++) {
                        for (iFilterCol = 0; iFilterCol < filterSize; iFilterCol++) {
                            value += data[(iRow*stride + iFilterRow) * dataSize + iCol*stride + iFilterCol + cLength * iChannel] * filters[(iFilterRow * filterSize + iFilterCol) + kLength * iChannel + iFilter * kcount];
                       }
                    }
                output[iFilter * (outputSize * outputSize) + iRow * outputSize + iCol] = (value);
            }
        }
    }
}

void pool2d(float *data, int dataSize, int nChannels,
            int poolSize, float *output) {
    int iChannel, iRow, iCol, iPoolRow, iPoolCol;

    int outputSize = dataSize + 1 - poolSize;
    float maxValue, thisValue;

    for (iChannel = 0; iChannel < nChannels; iChannel++) {
        for (iRow = 0; iRow <  outputSize; iRow++) {
            for (iCol = 0; iCol < outputSize; iCol++) {
                maxValue = 0;
                for (iPoolRow = 0; iPoolRow < poolSize; iPoolRow++) {
                    for (iPoolCol = 0; iPoolCol < poolSize; iPoolCol++) {
                        // data[iChannel][iRow + iPoolRow][iCol + iPoolCol]
                        thisValue = data[iChannel * (dataSize * dataSize) + (iRow + iPoolRow) * dataSize + iCol + iPoolCol];
                        if (thisValue > maxValue)
                            maxValue = thisValue;
                    }
                }
                output[iChannel * (outputSize * outputSize) + iRow * outputSize + iCol] = maxValue;
            }
        }
    }
}
            
void pool2d_backprop(float *convData, int dataSize,
             float *dNextLayer, int nChannels, int poolSize,
            float *convErrors) {
    //Important: assumes convData contains only zeroes

    int iChannel, iRow, iCol, iPoolRow, iPoolCol;

    int outputSize = dataSize + 1 - poolSize; //Note: outputSize is the size of the pooling output. In BP, this is the "dNextLayer"
    float maxValue, thisValue;
    int maxIndexR, maxIndexC;

    for (iChannel = 0; iChannel < nChannels; iChannel++) {
        for (iRow = 0; iRow <  outputSize; iRow++) {
            for (iCol = 0; iCol < outputSize; iCol++) {
                maxValue = -1;
                maxIndexR = maxIndexC = -1;
                for (iPoolRow = 0; iPoolRow < poolSize; iPoolRow++) {
                    for (iPoolCol = 0; iPoolCol < poolSize; iPoolCol++) {
                        // data[iChannel][iRow + iPoolRow][iCol + iPoolCol]
                        thisValue = convData[iChannel * (dataSize * dataSize) + (iRow + iPoolRow) * dataSize + iCol + iPoolCol];
                        if (thisValue > maxValue) {
                            maxValue = thisValue;
                            maxIndexR = iRow + iPoolRow;
                            maxIndexC = iCol + iPoolCol;
                        }
                    }
                }
                convErrors[iChannel * (dataSize * dataSize) + maxIndexR * dataSize + maxIndexC]  = dNextLayer[iChannel * (outputSize * outputSize) + iRow * outputSize + iCol];
            }
        }
    }
}
            

