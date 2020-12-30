#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <string.h>

// double *populateArrayData(int dimension) {
//    double *array = malloc(sizeof(double) * (unsigned) (dimension * dimension));
//    srand((unsigned int) time(NULL));

//    for (int i = 0; i < dimension; i++) {
//       for (int j = 0; j < dimension; j++) {
//             array[dimension * i + j] = (double) rand()/(float) (RAND_MAX/3);
//       }
//    }
//    return array;
// }

void printArray(double *arr, int dimension) {
   for (int i = 0; i < dimension; i++) {
      for (int j = 0; j < dimension; j++) {
         printf("%f, ", arr[dimension * i + j]);
      }
      printf("\n");
   }
}

double *calculateAverage(double *toAverage, int numRowsToAverage, int numColsToAverage, double precision) {
    double *toAverageCopy = malloc(sizeof(double) * (unsigned) (numRowsToAverage * numColsToAverage));
    memcpy(toAverageCopy, toAverage, sizeof(double) * (unsigned) (numRowsToAverage * numColsToAverage));

    for (int i=1; i<numRowsToAverage - 1; i++) {
        for (int j=1; j<numColsToAverage - 1; j++) {
            double newAverage = (
                toAverageCopy[numColsToAverage * (i-1) + j] +
                toAverageCopy[numColsToAverage * (i+1) + j] +
                toAverageCopy[numColsToAverage * i + (j-1)] +
                toAverageCopy[numColsToAverage * i + (j+1)]
            )/4;

            if (fabs(toAverageCopy[numColsToAverage * i + j] - newAverage) > precision) {
                toAverage[numColsToAverage * i + j] = newAverage;
            }
        }
    }
    free(toAverageCopy);
    return toAverage;
}

void distributeRowIndexesToProccesors(int *rowSplitPerProcessor, int dimension, int totalProcs){
    for (int i = 0; i < totalProcs; i++) {
        rowSplitPerProcessor[i] = (dimension - 2) / totalProcs;
    }

    int remainder = (dimension - 2) % totalProcs;
    if (remainder > 0) {
        for (int i = 0; i < remainder; i++) {
            rowSplitPerProcessor[i]++;
        }
    }
}


bool sendAndAverageRows(double *readArray, int *rowSplitPerProcessor, int numOfProcessors, bool precisionNotReached, int dimension, double precision, MPI_Status status) {
    int rootProcess = 0, rowsSentToProcessor = 0, rowsRecievedFromProcessor = rowSplitPerProcessor[rootProcess] + 1;
    precisionNotReached = true;

    // Get data ready to process
    // If we have processors to use, send data away
    // And do some processing in root.
    for (int rank = 0; rank < numOfProcessors; rank++) {
        int rowIndexesToProcessor = rowSplitPerProcessor[rank];
        int startRow = rowsSentToProcessor;
        int endRow = rowsSentToProcessor + rowIndexesToProcessor + 2;

        // Prepare a section of the readArray to process/send to other processors.
        double *rowsToProcess = malloc(sizeof(double) * (unsigned) ((rowIndexesToProcessor + 2) * dimension));
        for (int i = startRow; i<endRow; i++) {
            for (int j = 0; j<dimension; j++) {
                rowsToProcess[dimension * (i-startRow) + j] = readArray[dimension * i + j];
            }
        }
        rowsSentToProcessor += rowIndexesToProcessor;

        // If we're the root, do the processing.
        if (rank == rootProcess) {
            rowsToProcess = calculateAverage(rowsToProcess, endRow - startRow, dimension, precision);

            for (int i=startRow+1; i<endRow-1; i++) {
                for (int j=1; j<dimension-1; j++) {
                    // Update readArray if we're not in precision.
                    if (fabs(readArray[dimension * i + j] - rowsToProcess[dimension * (i-startRow) + j]) > precision) {
                        readArray[dimension * i + j] = rowsToProcess[dimension * (i-startRow) + j];
                        precisionNotReached = false;
                    }
                }
            }
        }
        // If we're not, send the rows off to each individual processor available
        else {
            int numberOfElementsToProcess = (rowIndexesToProcessor + 2) * dimension;
            MPI_Send(rowsToProcess, numberOfElementsToProcess, MPI_DOUBLE, rank, rank, MPI_COMM_WORLD);
        }

        free(rowsToProcess);
    }

    for (int rank = 1; rank < numOfProcessors; rank++) {
        int rowIndexesToReceive = rowSplitPerProcessor[rank];
        int numberOfElementsToProcess = (rowIndexesToReceive + 2) * dimension;
        double *averagedValues = malloc(sizeof(double) * (unsigned) ((rowIndexesToReceive + 2) * dimension));

        // Receive data from all the processors
        MPI_Recv(averagedValues, numberOfElementsToProcess, MPI_DOUBLE, rank, rootProcess, MPI_COMM_WORLD, &status);
        int endRow = rowsRecievedFromProcessor + rowIndexesToReceive;

        for (int i=rowsRecievedFromProcessor; i<endRow; i++) {
            for (int j=1; j<dimension-1; j++) {
                // Check if the values received from processors are within precision
                // If not, update readArray and start the while loop again.
                if (fabs(averagedValues[dimension * (i-rowsRecievedFromProcessor+1) + j] - readArray[dimension * i + j]) > precision) {
                    readArray[dimension * i + j] = averagedValues[dimension * (i-rowsRecievedFromProcessor+1) + j];
                    precisionNotReached = false;
                }
            }
        }
        rowsRecievedFromProcessor += rowIndexesToReceive;

        free(averagedValues);
    }

    // if (precisionNotReached) {
    //     printf("------------------FINAL ARRAY:------------------\n");
    //     printArray(readArray, dimension);
    // }

    return precisionNotReached;
}

/*
* Each processor will receive rows from the root to process
* Each processor will then individually average the data it has been given and then send it back to the root.
* Each processor will run this function on it at the same time as the other processors.
*/
void receiveAndAverageRows(int *rowSplitPerProcessor, int currentRank, int dimension, double precision, MPI_Status status) {
    int rowsToRecieve = rowSplitPerProcessor[currentRank] + 2;
    int numberOfElementsToProcess = rowsToRecieve * dimension;

    double *toAverage = malloc(sizeof(double) * (unsigned) (numberOfElementsToProcess));
    MPI_Recv(toAverage, numberOfElementsToProcess, MPI_DOUBLE, 0, currentRank, MPI_COMM_WORLD, &status);  

    toAverage = calculateAverage(toAverage, rowsToRecieve, dimension, precision);

    MPI_Send(toAverage, numberOfElementsToProcess, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    free(toAverage);
}

void testIt(double *readArray, int dimension, double precision) {
    int numOfProcessors, currentRank, rootProcess = 0;
    double start, end;
    bool precisionNotReached = false;
    MPI_Status status;

    int rc = MPI_Init(NULL, NULL);
    if (rc != MPI_SUCCESS) {
        printf ("Error starting MPI program\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

   // Get current processor ID and number of processors available
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcessors);

    printf("Processors: %d\nDimension: %d\n", numOfProcessors, dimension);

    int *rowSplitPerProcessor = malloc(sizeof(int) * (unsigned) (numOfProcessors));
    distributeRowIndexesToProccesors(rowSplitPerProcessor, dimension, numOfProcessors);

    // double *readArray = populateArrayData(dimension);
    
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    start = MPI_Wtime();
    do {
        if (currentRank == rootProcess) {
            precisionNotReached = sendAndAverageRows(
                readArray, 
                rowSplitPerProcessor, 
                numOfProcessors, 
                precisionNotReached, 
                dimension, 
                precision,
                status
            );
        } else {
            receiveAndAverageRows(
                rowSplitPerProcessor, 
                currentRank, 
                dimension, 
                precision, 
                status
            );
        }
        // Tell all processors whether they can stop processing or not
        MPI_Bcast(&precisionNotReached, 1, MPI_C_BOOL, rootProcess, MPI_COMM_WORLD);
    } while (!precisionNotReached);

    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    end = MPI_Wtime();
    if (currentRank == 0) { /* use time on master node */
        printf("Runtime = %f\n", end-start);
    }

    free(rowSplitPerProcessor);
    MPI_Finalize();
}

int main(int argc, char **argv) {
    char inputFilename[32];

    // default, will be updated
    int dimension = 0;
    double precision = 0.01;

    if (argc <= 1) {
        printf("\nNo arguments found. Something's gone wrong.\n");
        return -1;
    }
    if (argc >= 2) {
        precision = strtod(argv[1], NULL);
    }
    if (argc >= 3) {
        strcpy(inputFilename, argv[2]);
    }

    FILE *fp;
    char dimStr[6]; // max dim = 99,999 + termination char
    fp = fopen(inputFilename, "r");
    if (fp == NULL) {
        perror("Error opening file");
        return -1;
    }

    // read in array dimension
    if (fgets(dimStr, 6, fp) != NULL) {
        dimension = atoi(dimStr);
    }
    if (dimension < 0) {
        printf("\n[ERROR] Dimension cannot be zero.\n");
        return 1;
    }

    unsigned long arrSize = sizeof(double) * (unsigned) (dimension * dimension);
    double *readArray = malloc(arrSize);

    // read in input array
    for (int i=0; i<dimension; i++) {
        for (int j=0; j<dimension; j++) {
            char read[32];
            fscanf(fp, " %s", read);
            readArray[dimension * i + j] = strtod(read, NULL);
        }
    }
    fclose(fp);

    testIt(readArray, dimension, precision);

    free(readArray);
    return 0;
}