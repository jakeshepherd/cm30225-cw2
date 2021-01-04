#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

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
        rowSplitPerProcessor[i] = (dimension - 2) / useableProcs;
    }

    int remainder = (dimension - 2) % useableProcs;
    if (remainder > 0) {
        // evenly distribute the remainder rows
        for (int i = 1; i <= remainder; i++) {
            rowSplitPerProcessor[i]++;
        }
    }
}

/*
 * Each processor will run this function individually on the data that it has to average it
 */
void calculateAverage(double *oldAverages, int numberOfRowsToAverage, int dimension, double prec, bool *processorDataConverged) {
    double *temp = malloc(sizeof(double) * (unsigned) (numberOfRowsToAverage * dimension));
    memcpy(temp, oldAverages, sizeof(double) * (unsigned) (numberOfRowsToAverage * dimension));
    *processorDataConverged = true;

    for (int i = 1; i < numberOfRowsToAverage - 1; i++) {
        for (int j = 1; j < dimension - 1; j++) {
            double average = (
                temp[dimension * (i + 1) + j] +
                temp[dimension * (i - 1) + j] +
                temp[dimension * i + (j + 1)] +
                temp[dimension * i + (j - 1)]
            ) / 4.0;

            if (fabs(temp[dimension * i + j] - average) > prec) {
                oldAverages[dimension * i + j] = average;
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
    int rootProcessor = 0, numberOfIterations = 0, numberOfBoundaryRows = 2, numberOfRowsForProcessor, elementsForProcessor;
    double *updatedRows, *dataPerProcessor = NULL;
    bool precisionNotReached = false;

    // find number of rows to process per processor
    int *rowSplitPerProcessor = malloc(sizeof(int) * (unsigned) (numOfProcessors));
    distributeRowIndexesToProccesors(rowSplitPerProcessor, dimension, numOfProcessors);

    // Start timing
    // Barrier here to make sure that all the processors available are all ready to go
    MPI_Barrier(MPI_COMM_WORLD);
    runtime = MPI_Wtime();

    do { // iterate until converged
        if (currentRank == rootProcessor) {
            // If this is the first time going round
            // send every element of the data that needs to be processed
            if (numberOfIterations == 0) {
                int totalElementsSent = 0;

                for (int i = 1; i < numOfProcessors; i++) {
                    int numRowsToSend = rowSplitPerProcessor[i] + numberOfBoundaryRows;
                    int elementsToSend = numRowsToSend * dimension;

                    MPI_Send(&testValues[totalElementsSent], elementsToSend, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    totalElementsSent += elementsToSend - (numberOfBoundaryRows * dimension);
                }
            } 
            // Else, only send the boundaryRowsPerProcessor of the data that needs to be processed
            else {
                // send boundaryRowsPerProcessor
                double *boundaryRowsPerProcessor = malloc(sizeof(double) * (unsigned) (numberOfBoundaryRows * dimension));
                int rowsSent = 0; // the total rows covered by the sends

                for (int i = 1; i < numOfProcessors; i++) {
                    int upperBoundary = rowsSent;
                    int lowerBoundary = upperBoundary + rowSplitPerProcessor[i] + 1;

                    for (int j = 0; j < dimension; j++) {
                        boundaryRowsPerProcessor[dimension * 0 + j] = testValues[dimension * upperBoundary + j];
                        boundaryRowsPerProcessor[dimension * 1 + j] = testValues[dimension * lowerBoundary + j];
                    }

                    MPI_Send(boundaryRowsPerProcessor, (numberOfBoundaryRows * dimension), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    rowsSent += rowSplitPerProcessor[i];
                }
                free(boundaryRowsPerProcessor);
            }
            precisionNotReached = true;
            bool processorDataConverged = false;

            // receive chunks are converged? & updated precisionNotReached
            // precisionNotReached = true if all chunks are converged
            // check whether chunks have hit precision, if any single dataPerProcessor hasn't, withinPrecision is false
            for (int i = 1; i < numOfProcessors; i++) {
                // receive bool indicating dataPerProcessor convergence from each child processor
                MPI_Recv(&processorDataConverged, 1, MPI_C_BOOL, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (!processorDataConverged) {
                    precisionNotReached = false;
                }
            }

            // broadcast array convergence to child processors
            MPI_Bcast(&precisionNotReached, 1, MPI_C_BOOL, rootProcessor, MPI_COMM_WORLD);

            if (precisionNotReached) {
                // rec chunks & merge to main array
                int totalRowsReceived = 1;
                for (int i = 1; i < numOfProcessors; i++) {
                    numberOfRowsForProcessor = rowSplitPerProcessor[i] + numberOfBoundaryRows;
                    // size of the dataPerProcessor in terms of doubles, without the buffers
                    elementsForProcessor = numberOfRowsForProcessor * dimension;

                    updatedRows = malloc(sizeof(double) * (unsigned) elementsForProcessor);
                    MPI_Recv(updatedRows, elementsForProcessor, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // merge the averaged dataPerProcessor into the main array
                    for (int i = 0; i < numberOfRowsForProcessor - numberOfBoundaryRows; i++) {
                        for (int j = 1; j < dimension - 1; j++) {
                            testValues[dimension * (totalRowsReceived + i) + j] = updatedRows[dimension * (i+1) + j];
                        }
                    }
                    totalRowsReceived += numberOfRowsForProcessor - numberOfBoundaryRows;
                    free(updatedRows);
                }
            } else {
                // rec boundaryRowsPerProcessor & merge to main array
                double *boundaryRowsPerProcessor = malloc(sizeof(double) * (unsigned) (numberOfBoundaryRows * dimension));
                int firstIndex, lastIndex = 0;

                for (int i = 1; i < numOfProcessors; i++) { 
                    MPI_Recv(boundaryRowsPerProcessor, (numberOfBoundaryRows * dimension), MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    firstIndex = lastIndex + 1;
                    lastIndex = firstIndex + rowSplitPerProcessor[i] - 1;

                    for (int j = 1; j < dimension -1; j++) {
                        testValues[firstIndex * dimension + j] = boundaryRowsPerProcessor[dimension * 0 + j];
                        testValues[lastIndex * dimension + j] = boundaryRowsPerProcessor[dimension * 1 + j];
                    }

                }

                free(boundaryRowsPerProcessor);
            }
        } else {
            if (numberOfIterations == 0) {
                // rec chunks and store
                numberOfRowsForProcessor = rowSplitPerProcessor[currentRank] + numberOfBoundaryRows;
                elementsForProcessor = numberOfRowsForProcessor * dimension;

                // memory for received dataPerProcessor
                dataPerProcessor = malloc(sizeof(double) * (unsigned) (numberOfRowsForProcessor * dimension));

                MPI_Recv(dataPerProcessor, elementsForProcessor, MPI_DOUBLE, rootProcessor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // rec boundaryRowsPerProcessor and store
                double *boundaryRowsPerProcessor = malloc(sizeof(double) * (unsigned) (numberOfBoundaryRows * dimension));
                MPI_Recv(boundaryRowsPerProcessor, (numberOfBoundaryRows * dimension), MPI_DOUBLE, rootProcessor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //set first row of portion to first border
                for (int i = 0; i < dimension; i++) {
                    dataPerProcessor[i] = boundaryRowsPerProcessor[i];
                    dataPerProcessor[dimension * (numberOfRowsForProcessor - 1) + i] = boundaryRowsPerProcessor[dimension + i];
                }

                free(boundaryRowsPerProcessor);
            }

            // average chunks and check converged
            bool processorDataConverged = false;
            calculateAverage(dataPerProcessor, numberOfRowsForProcessor, dimension, prec, &processorDataConverged);

            // send dataPerProcessor converged to root
            MPI_Send(&processorDataConverged, 1, MPI_C_BOOL, rootProcessor, 2, MPI_COMM_WORLD);

            // receive broadcast for (all array) precisionNotReached
            // receive withinPrecision bool from the master process, is a blocking call so also acts as a synchronise
            MPI_Bcast(&precisionNotReached, 1, MPI_C_BOOL, rootProcessor, MPI_COMM_WORLD);

            if (precisionNotReached) {
                // send updated chunks to root
                // send back to the master process only modified inner rows of the dataPerProcessor

                MPI_Send(dataPerProcessor, elementsForProcessor, MPI_DOUBLE, rootProcessor, 1, MPI_COMM_WORLD);
            } else {
                // send boundaryRowsPerProcessor to root
                double *boundaryRowsPerProcessor = malloc(sizeof(double) * (unsigned) (numberOfBoundaryRows * dimension));

                for (int j = 0; j < dimension; j++) {
                    boundaryRowsPerProcessor[dimension * 0 + j] = dataPerProcessor[dimension * 1 + j];
                    boundaryRowsPerProcessor[dimension * 1 + j] = dataPerProcessor[dimension * (numberOfRowsForProcessor - numberOfBoundaryRows) + j];
                }

                MPI_Send(boundaryRowsPerProcessor, (numberOfBoundaryRows * dimension), MPI_DOUBLE, rootProcessor, 3, MPI_COMM_WORLD);
                free(boundaryRowsPerProcessor);
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