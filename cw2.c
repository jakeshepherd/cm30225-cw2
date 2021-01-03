#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define NUM_OF_BOUNDARY_ROWS 2

void printArray(double *arr, int dimension) {
   for (int i = 0; i < dimension; i++) {
      for (int j = 0; j < dimension; j++) {
         printf("%f, ", arr[dimension * i + j]);
      }
      printf("\n");
   }
}

/*
 * Function: distributeRowIndexesToProccesors
 * ----------------------------
 * Allocate an even distribution of rows to available processors. The 
 * available processors is equal to the total processors - 1. If the 
 * problem size is not evenly divisable, the remainders will be spread
 * over the first n processors, with n equal to the remainder rows.
 * 
 * rowSplitPerProcessor: int array to store the result
 * dimension: dimension of the problem array
 * numOfProcessors: total processors being used
 */
void distributeRowIndexesToProccesors(int *rowSplitPerProcessor, int dimension, int numOfProcessors) {
    // accounting for no work in root
    int useableProcs = numOfProcessors - 1;

    rowSplitPerProcessor[0] = 0;
    for (int i = 1; i <= useableProcs; i++) {
        rowSplitPerProcessor[i] = (dimension - NUM_OF_BOUNDARY_ROWS) / useableProcs;
    }

    int remainder = (dimension - NUM_OF_BOUNDARY_ROWS) % useableProcs;
    if (remainder > 0) {
        // evenly distribute the remainder rows
        for (int i = 1; i <= remainder; i++) {
            rowSplitPerProcessor[i]++;
        }
    }
}

void averageRows(double *readArr, int numRows, int dimension, double prec, bool *processorDataConverged) {
    double *temp = malloc(sizeof(double) * (unsigned) (numRows * dimension));
    memcpy(temp, readArr, sizeof(double) * (unsigned) (numRows * dimension));
    *processorDataConverged = true;

    for (int i = 1; i < numRows - 1; i++) {
        for (int j = 1; j < dimension - 1; j++) {
            double avg = (
                temp[dimension * (i + 1) + j] +
                temp[dimension * (i - 1) + j] + 
                temp[dimension * i + (j + 1)] + 
                temp[dimension * i + (j - 1)]
                ) / 4.0;

            if (fabs(avg - temp[dimension * i + j]) > prec) {
                readArr[dimension * i + j] = avg;
                *processorDataConverged = false;
            }
        }
    }

    free(temp);
}

/*
 * Function: testIt
 * ----------------------------
 * Runs the solver and iterates until the problem has converged.
 * 
 * testValues: starting array, will be an empty pointer in all processors except 
 *       the root
 * dimension: dimension of the problem array 
 * prec: precision at which the problem has converged 
 * currentRank: rank of the current processor 
 * numOfProcessors: total number of processors being used
 */
void testIt(double *testValues, int dimension, double prec, int currentRank, int numOfProcessors) {
    double runtime, avgRuntime;
    int rootProcessor = 0, numberOfIterations = 0;
    bool precisionNotReached = false;

    // find number of rows to process per processor
    int *rowSplitPerProcessor = malloc(sizeof(int) * (unsigned) (numOfProcessors));
    distributeRowIndexesToProccesors(rowSplitPerProcessor, dimension, numOfProcessors);

    int rowsInChunk, elemsInChunk;
    double *updatedRows;
    double *dataPerProcessor = NULL;

    // make sure all nodes are ready to work before timing
    MPI_Barrier(MPI_COMM_WORLD);
    runtime = MPI_Wtime();

    do { // iterate until converged
        if (currentRank == rootProcessor) {
            if (numberOfIterations == 0) {
                // send chunks to each processor
                int totalElementsSent = 0;

                for (int i = 1; i < numOfProcessors; i++) {
                    int numRowsToSend = rowSplitPerProcessor[i] + NUM_OF_BOUNDARY_ROWS;
                    int elementsToSend = numRowsToSend * dimension;

                    MPI_Send(&testValues[totalElementsSent], elementsToSend, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    totalElementsSent += elementsToSend - (NUM_OF_BOUNDARY_ROWS * dimension);
                }
            } else {
                // send boundaries
                double *boundaries = malloc(sizeof(double) * (unsigned) (NUM_OF_BOUNDARY_ROWS * dimension));
                int rowsSent = 0; // the total rows covered by the sends

                for (int i = 1; i < numOfProcessors; i++) {
                    int firstBoundaryRow = rowsSent;
                    int lastBoundaryRow = firstBoundaryRow + rowSplitPerProcessor[i] + 1;

                    for (int j = 0; j < dimension; j++) {
                        boundaries[dimension * 0 + j] = testValues[dimension * firstBoundaryRow + j];
                        boundaries[dimension * 1 + j] = testValues[dimension * lastBoundaryRow + j];
                    }

                    MPI_Send(boundaries, (NUM_OF_BOUNDARY_ROWS * dimension), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    rowsSent += rowSplitPerProcessor[i];
                }
                free(boundaries);
            }
            precisionNotReached = true;
            bool chunkHasConverged = false;

            // receive chunks are converged? & updated precisionNotReached
            // precisionNotReached = true if all chunks are converged
            // check whether chunks have hit precision, if any single dataPerProcessor hasn't, withinPrecision is false
            for (int i = 1; i < numOfProcessors; i++) {
                // receive bool indicating dataPerProcessor convergence from each child processor
                MPI_Recv(&chunkHasConverged, 1, MPI_C_BOOL, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (!chunkHasConverged) {
                    precisionNotReached = false;
                }
            }

            // broadcast array convergence to child processors
            MPI_Bcast(&precisionNotReached, 1, MPI_C_BOOL, rootProcessor, MPI_COMM_WORLD);

            if (precisionNotReached) {
                // rec chunks & merge to main array
                int totalRowsReceived= 1;
                for (int i = 1; i < numOfProcessors; i++) {
                    rowsInChunk = rowSplitPerProcessor[i] + NUM_OF_BOUNDARY_ROWS;
                    // size of the dataPerProcessor in terms of doubles, without the buffers
                    elemsInChunk = rowsInChunk * dimension;

                    updatedRows = malloc(sizeof(double) * (unsigned) elemsInChunk);
                    MPI_Recv(updatedRows, elemsInChunk, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // merge the averaged dataPerProcessor into the main array
                    for (int i = 0; i < rowsInChunk - NUM_OF_BOUNDARY_ROWS; i++) {
                        for (int j = 1; j < dimension - 1; j++) {
                            testValues[dimension * (totalRowsReceived + i) + j] = updatedRows[dimension * (i+1) + j];
                        }
                    }
                    totalRowsReceived += rowsInChunk - NUM_OF_BOUNDARY_ROWS;
                    free(updatedRows);
                }
            } else {
                // rec boundaries & merge to main array
                double *boundaries = malloc(sizeof(double) * (unsigned) (NUM_OF_BOUNDARY_ROWS * dimension));
                int firstIndex, lastIndex = 0;

                for (int i = 1; i < numOfProcessors; i++) { 
                    MPI_Recv(boundaries, (NUM_OF_BOUNDARY_ROWS * dimension), MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    firstIndex = lastIndex + 1;
                    lastIndex = firstIndex + rowSplitPerProcessor[i] - 1;

                    for (int j = 1; j < dimension -1; j++) {
                        testValues[firstIndex * dimension + j] = boundaries[dimension * 0 + j];
                        testValues[lastIndex * dimension + j] = boundaries[dimension * 1 + j];
                    }

                }

                free(boundaries);
            }
        } else {
            if (numberOfIterations == 0) {
                // rec chunks and store
                rowsInChunk = rowSplitPerProcessor[currentRank] + NUM_OF_BOUNDARY_ROWS;
                elemsInChunk = rowsInChunk * dimension;

                // memory for received dataPerProcessor
                dataPerProcessor = malloc(sizeof(double) * (unsigned) (rowsInChunk * dimension));

                MPI_Recv(dataPerProcessor, elemsInChunk, MPI_DOUBLE, rootProcessor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // rec boundaries and store
                double *boundaries = malloc(sizeof(double) * (unsigned) (NUM_OF_BOUNDARY_ROWS * dimension));
                MPI_Recv(boundaries, (NUM_OF_BOUNDARY_ROWS * dimension), MPI_DOUBLE, rootProcessor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //set first row of portion to first border
                for (int i = 0; i < dimension; i++) {
                    dataPerProcessor[i] = boundaries[i];
                    dataPerProcessor[dimension * (rowsInChunk - 1) + i] = boundaries[dimension + i];
                }

                free(boundaries);
            }

            // avg chunks and check converged
            bool processorDataConverged = false;
            averageRows(dataPerProcessor, rowsInChunk, dimension, prec, &processorDataConverged);

            // send dataPerProcessor converged to root
            MPI_Send(&processorDataConverged, 1, MPI_C_BOOL, rootProcessor, 2, MPI_COMM_WORLD);

            // receive broadcast for (all array) precisionNotReached
            // receive withinPrecision bool from the master process, is a blocking call so also acts as a synchronise
            MPI_Bcast(&precisionNotReached, 1, MPI_C_BOOL, rootProcessor, MPI_COMM_WORLD);

            if (precisionNotReached) {
                // send updated chunks to root
                // send back to the master process only modified inner rows of the dataPerProcessor

                MPI_Send(dataPerProcessor, elemsInChunk, MPI_DOUBLE, rootProcessor, 1, MPI_COMM_WORLD);
            } else {
                // send boundaries to root
                double *boundaries = malloc(sizeof(double) * (unsigned) (NUM_OF_BOUNDARY_ROWS * dimension));

                for (int j = 0; j < dimension; j++) {
                    boundaries[dimension * 0 + j] = dataPerProcessor[dimension * 1 + j];
                    boundaries[dimension * 1 + j] = dataPerProcessor[dimension * (rowsInChunk - NUM_OF_BOUNDARY_ROWS) + j];
                }

                MPI_Send(boundaries, (NUM_OF_BOUNDARY_ROWS * dimension), MPI_DOUBLE, rootProcessor, 3, MPI_COMM_WORLD);
                free(boundaries);
            }
        }

        printf("it: %d\n", numberOfIterations);
        numberOfIterations++;
    } while (!precisionNotReached);

    // Barrier here to make sure that every processor has finished it's work
    // This is so that we do not start calculating the runtime before the work has been done
    // We can then stop the timer and work out the runtime.
    MPI_Barrier(MPI_COMM_WORLD);
    runtime = MPI_Wtime() - runtime;

    // reduce (sum) all processor runtimes into a single value in the root
    MPI_Reduce(&runtime, &avgRuntime, 1, MPI_DOUBLE, MPI_SUM, rootProcessor, MPI_COMM_WORLD);

    // and get the average runtime
    if (currentRank == rootProcessor) {
        avgRuntime /= numOfProcessors;
        printf("-------------------\nRuntime: %f\n", avgRuntime);
    }

    if (currentRank == rootProcessor) {
        printf("\n\nFINAL ARRAY:\n");
        printArray(testValues, dimension);
    }

    free(rowSplitPerProcessor);
    if (dataPerProcessor != NULL) {
        free(dataPerProcessor);
    }
}

int main(int argc, char *argv[]) {
    int currentRank, numOfProcessors, dimension = 0, rootProcessor = 0;
    double precision = 0.01;
    double *testValues = NULL;
    char inputFilename[32];

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcessors);

    if (numOfProcessors < 2) {
        printf("-- There must be at least 2 processors.\n");
        exit(-1);
    }

    // process command args
    if (argc <= 1) {
        printf("\nNo arguments found. Something's gone wrong.\n");
        return -1;
    }
    if (argc >= 2) {
        dimension = atoi(argv[1]);
    }
    if (argc >= 3) {
        precision = strtod(argv[2], NULL);
    }
    if (argc >= 4) {
        strcpy(inputFilename, argv[3]);
    }

    // read in data in root
    if (currentRank == rootProcessor) {
        FILE *fp;
        char dimStr[6]; // max dimension = 99,999 + termination char
        fp = fopen(inputFilename, "r");
        if (fp == NULL) {
            perror("Error opening file");
            return -1;
        }

        // read in array dim 
        if (fgets(dimStr, 6, fp) != NULL) {
            dimension = atoi(dimStr);
        }
        if (dimension < 0) {
            printf("\n[ERROR] dimension cannot be zero.\n");
            return 1;
        }

        // total size of the full problem array
        unsigned long arrSize = sizeof(double) * (unsigned) (dimension * dimension);
        testValues = malloc(arrSize);

        // read in input array
        for (int i=0; i<dimension; i++) {
            for (int j=0; j<dimension; j++) {
                char read[32];
                fscanf(fp, " %s", read);
                testValues[dimension * i + j] = strtod(read, NULL);
            }
        }
        fclose(fp);
    }

    testIt(testValues, dimension, precision, currentRank, numOfProcessors);

    MPI_Finalize();
    if (testValues != NULL) {
        free(testValues);
    }

    return 0;
}